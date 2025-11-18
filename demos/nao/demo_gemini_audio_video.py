import asyncio
import base64
import io
import time
import contextlib  # needed for suppress

import cv2
from PIL import Image

from google import genai
from google.genai import types
from sic_framework.devices.nao import NaoqiTextToSpeechRequest

from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging

from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_camera import NaoqiCameraConf
from sic_framework.core.message_python2 import AudioMessage, CompressedImageMessage


class NaoGeminiConversation(SICApplication):
    """
    NAO Gemini Live TEXT conversation with tool calling, audio, and live camera feed.

    - Streams NAO microphone audio to Gemini Live.
    - Streams NAO camera frames to Gemini Live (throttled).
    - Shows exactly the frames that are sent to Gemini in an OpenCV window.
    - Logs Live API usage and cost via usage_metadata.
    - Periodically triggers short automatic commentary turns.
    """

    INPUT_TEXT_PRICE_PER_MTOK = 0.50
    INPUT_MEDIA_PRICE_PER_MTOK = 3.00  # audio + images
    OUTPUT_TEXT_PRICE_PER_MTOK = 2.00

    def __init__(self):
        super(NaoGeminiConversation, self).__init__()

        self.nao_ip = "10.0.0.242"

        self.nao = None
        self.gemini_session = None
        self.loop = None

        self.is_nao_speaking = False
        self.model_is_speaking = False
        self.buffered_text = []

        # How often to send frames to Gemini (seconds)
        self.last_frame_sent_ts = 0.0
        self.frame_interval = 0.5

        # How often to trigger a commentary turn (seconds)
        # Tune this to control how frequently you get a 1-sentence line
        self.comment_interval = 1.0

        # Last frame that WAS ACTUALLY SENT to Gemini (RGB, upright)
        self.latest_image = None

        # Usage / cost tracking from usage_metadata
        self.total_input_text_tokens = 0
        self.total_input_audio_tokens = 0
        self.total_input_image_tokens = 0
        self.total_output_text_tokens = 0

        # Last cumulative totals seen
        self._last_prompt_text_total = 0
        self._last_prompt_audio_total = 0
        self._last_prompt_image_total = 0
        self._last_response_text_total = 0

        self.set_log_level(sic_logging.INFO)
        self.setup()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    def setup(self):
        self.logger.info("Initializing NAO...")

        camera_conf = NaoqiCameraConf(vflip=1)
        self.nao = Nao(ip=self.nao_ip, top_camera_conf=camera_conf)

        self.logger.info("Registering NAO top camera callback...")
        self.nao.top_camera.register_callback(self.on_nao_image)

    # -------------------------------------------------------------------------
    # Usage / cost helpers
    # -------------------------------------------------------------------------
    def _update_usage_from_metadata(self, meta):
        if meta is None:
            return

        if meta.prompt_tokens_details:
            curr_text = curr_audio = curr_image = 0
            for d in meta.prompt_tokens_details:
                mod = getattr(d.modality, "value", str(d.modality))
                if mod == "TEXT":
                    curr_text += d.token_count
                elif mod == "AUDIO":
                    curr_audio += d.token_count
                elif mod == "IMAGE":
                    curr_image += d.token_count

            dt = curr_text - self._last_prompt_text_total
            da = curr_audio - self._last_prompt_audio_total
            di = curr_image - self._last_prompt_image_total

            if dt > 0:
                self.total_input_text_tokens += dt
            if da > 0:
                self.total_input_audio_tokens += da
            if di > 0:
                self.total_input_image_tokens += di

            self._last_prompt_text_total = curr_text
            self._last_prompt_audio_total = curr_audio
            self._last_prompt_image_total = curr_image

        if meta.response_tokens_details:
            curr_resp_text = 0
            for d in meta.response_tokens_details:
                mod = getattr(d.modality, "value", str(d.modality))
                if mod == "TEXT":
                    curr_resp_text += d.token_count

            dr = curr_resp_text - self._last_response_text_total
            if dr > 0:
                self.total_output_text_tokens += dr
            self._last_response_text_total = curr_resp_text

    def _log_costs(self, context: str = ""):
        media_tokens = self.total_input_audio_tokens + self.total_input_image_tokens

        input_text_cost = (
            self.total_input_text_tokens / 1_000_000.0
        ) * self.INPUT_TEXT_PRICE_PER_MTOK
        input_media_cost = (
            media_tokens / 1_000_000.0
        ) * self.INPUT_MEDIA_PRICE_PER_MTOK
        output_text_cost = (
            self.total_output_text_tokens / 1_000_000.0
        ) * self.OUTPUT_TEXT_PRICE_PER_MTOK

        total_cost = input_text_cost + input_media_cost + output_text_cost

        self.logger.info(
            "[COST] context=%s | in_text_tokens=%d in_audio_tokens=%d "
            "in_image_tokens=%d out_text_tokens=%d | "
            "input_text_cost≈$%.4f input_media_cost≈$%.4f "
            "output_text_cost≈$%.4f total≈$%.4f",
            context,
            self.total_input_text_tokens,
            self.total_input_audio_tokens,
            self.total_input_image_tokens,
            self.total_output_text_tokens,
            input_text_cost,
            input_media_cost,
            output_text_cost,
            total_cost,
        )

    # -------------------------------------------------------------------------
    # NAO-side actions
    # -------------------------------------------------------------------------
    async def perform_nao_dance(self, style: str | None = None):
        if style:
            self.logger.info(f"NAO dance requested with style: {style}")
        else:
            self.logger.info("NAO dance requested with default style")
        await asyncio.sleep(5)
        self.logger.info("NAO dance routine completed.")

    async def _speak_on_nao(self, text: str):
        """
        Run blocking NAO TTS in a thread so the asyncio event loop
        keeps running. Mic is blocked via is_nao_speaking.
        """
        loop = asyncio.get_running_loop()

        def _blocking_tts():
            self.nao.tts.request(
                NaoqiTextToSpeechRequest(text),
                block=True,
            )

        self.is_nao_speaking = True
        try:
            await loop.run_in_executor(None, _blocking_tts)
        finally:
            self.is_nao_speaking = False

    # -------------------------------------------------------------------------
    # Audio NAO → Gemini
    # -------------------------------------------------------------------------
    def on_nao_audio(self, message: AudioMessage):
        # Block mic if either the model is generating or NAO is speaking
        if self.is_nao_speaking or self.model_is_speaking:
            return

        data = message.waveform
        if not data:
            return

        if self.gemini_session and self.loop and not self.loop.is_closed():
            coro = self.gemini_session.send_realtime_input(
                audio=types.Blob(
                    data=data,
                    mime_type="audio/pcm;rate=16000",
                )
            )
            asyncio.run_coroutine_threadsafe(coro, self.loop)

    # -------------------------------------------------------------------------
    # Video NAO → Gemini + “only show what we send”
    # -------------------------------------------------------------------------
    def on_nao_image(self, image_message: CompressedImageMessage):
        """
        Called every camera frame.

        Only when the throttle condition passes do we:
        - rotate 180°
        - store in latest_image
        - encode JPEG
        - send to Gemini

        So the OpenCV window shows exactly what Gemini received.
        """
        if not (self.gemini_session and self.loop and not self.loop.is_closed()):
            return

        now = time.monotonic()
        if now - self.last_frame_sent_ts < self.frame_interval:
            return
        self.last_frame_sent_ts = now

        try:
            frame = image_message.image

            # Rotate 180°: flip vertical and horizontal
            frame = frame[::-1, ::-1]

            # Store only the sent frame for local display
            self.latest_image = frame

            img = Image.fromarray(frame)
            img.thumbnail((1024, 1024))

            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            image_bytes = buf.getvalue()

            self.logger.info(
                "Sending image frame to Gemini: size=%d bytes", len(image_bytes)
            )

            frame_payload = {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_bytes).decode("ascii"),
            }

            coro = self.gemini_session.send(input=frame_payload)
            asyncio.run_coroutine_threadsafe(coro, self.loop)

        except Exception as e:
            self.logger.error(f"Error in on_nao_image: {e}")

    # -------------------------------------------------------------------------
    # Display loop (OpenCV on main thread)
    # -------------------------------------------------------------------------
    def _show_latest_frame_if_any(self):
        if self.latest_image is None:
            return
        try:
            bgr_img = self.latest_image[..., ::-1]  # RGB -> BGR
            cv2.imshow("NAO Camera (sent to Gemini)", bgr_img)
            cv2.waitKey(1)
        except Exception as e:
            self.logger.error(f"Error in OpenCV display: {e}")

    async def display_loop(self):
        self.logger.info("OpenCV display loop started.")
        try:
            while not self.shutdown_event.is_set():
                self._show_latest_frame_if_any()
                await asyncio.sleep(0.03)  # ~30 FPS refresh
        except Exception as e:
            self.logger.error(f"Error in display_loop: {e}")
        finally:
            cv2.destroyAllWindows()
            self.logger.info("OpenCV display loop terminated.")

    # -------------------------------------------------------------------------
    # Automatic commentary loop
    # -------------------------------------------------------------------------
    async def commentary_loop(self):
        """
        Periodically forces an end-of-turn so the model produces a new
        very short commentary line based on the latest video/audio.

        It only triggers when neither the model nor NAO is currently speaking.
        """
        self.logger.info("Commentary loop started.")
        try:
            while not self.shutdown_event.is_set():
                await asyncio.sleep(self.comment_interval)

                # Only trigger a new turn if nothing is being said
                if (
                    self.gemini_session
                    and not self.model_is_speaking
                    and not self.is_nao_speaking
                ):
                    try:
                        # Dummy input to close the turn; model uses recent media
                        await self.gemini_session.send(
                            input=".",  # very small text
                            end_of_turn=True,
                        )
                    except Exception as e:
                        self.logger.error(f"Error in commentary_loop send: {e}")
        except Exception as e:
            self.logger.error(f"Error in commentary_loop: {e}")
        finally:
            self.logger.info("Commentary loop terminated.")

    # -------------------------------------------------------------------------
    # Tool-call handling
    # -------------------------------------------------------------------------
    async def handle_tool_calls(self, response):
        if not response.tool_call:
            return

        function_responses: list[types.FunctionResponse] = []

        for fc in response.tool_call.function_calls:
            name = fc.name
            args = dict(fc.args) if fc.args is not None else {}
            call_id = fc.id

            self.logger.info(f"Tool call: {name} args={args} id={call_id}")

            if name == "start_dance":
                style = args.get("style")
                await self.perform_nao_dance(style)
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id,
                        name=name,
                        response={"result": "ok", "style": style},
                    )
                )

        if function_responses:
            await self.gemini_session.send_tool_response(
                function_responses=function_responses
            )

    # -------------------------------------------------------------------------
    # Gemini Live main loop
    # -------------------------------------------------------------------------
    async def run_gemini(self):
        client = genai.Client()  # uses GOOGLE_API_KEY
        model = "gemini-live-2.5-flash-preview"

        start_dance_tool = {
            "name": "start_dance",
            "description": (
                "Make the NAO social robot perform a short dance routine. "
                "Call this whenever the user asks the robot to dance, "
                "do a dance, show a move, or similar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "style": {
                        "type": "string",
                        "description": (
                            "Optional style of dance, for example "
                            "'happy', 'slow', 'excited', 'silly', or "
                            "'robot dance'."
                        ),
                    },
                },
                "required": [],
            },
        }

        config = {
            "response_modalities": ["TEXT"],
            "system_instruction": (
                "You are Nao, a passionate football commentator robot. "
                "You receive a live video feed of a football match and audio input. "
                "Continuously commentate on the match in an engaging, "
                "enthusiastic manner. Each response you make must be a SINGLE, "
                "very short sentence, for example: "
                '"Blue player makes a pass.", '
                '"Team red receives a yellow card.", '
                '"A blue player scores a goal!" '
                "Always keep sentences this short and focused on the latest action."
            ),
            "tools": [
                {"function_declarations": [start_dance_tool]},
            ],
        }

        async with client.aio.live.connect(model=model, config=config) as session:
            self.gemini_session = session
            self.loop = asyncio.get_running_loop()

            # Start helper tasks
            display_task = asyncio.create_task(self.display_loop())
            commentary_task = asyncio.create_task(self.commentary_loop())

            self.nao.mic.register_callback(self.on_nao_audio)
            self.logger.info("Microphone callback registered. Start talking!")
            self.logger.info(
                "Camera callback registered. Streaming vision to Gemini and showing sent frames."
            )

            try:
                while not self.shutdown_event.is_set():
                    async for response in session.receive():
                        meta_data = response.usage_metadata
                        self.logger.info(f"Response metadata: {meta_data}")

                        if (
                            meta_data
                            and getattr(meta_data, "total_token_count", None)
                            is not None
                        ):
                            self._update_usage_from_metadata(meta_data)
                            self._log_costs(context="usage_metadata")

                        sc = response.server_content

                        if response.tool_call:
                            await self.handle_tool_calls(response)

                        if response.text is not None:
                            if not self.model_is_speaking:
                                self.model_is_speaking = True
                                # mic is blocked while model is generating
                                self.logger.info("Model started responding; mic muted.")
                            self.buffered_text.append(response.text)

                        if sc and sc.generation_complete:
                            full_text = "".join(self.buffered_text).strip()
                            self.buffered_text = []

                            if full_text:
                                self.logger.info(f"Full model response: {full_text}")
                                # speak asynchronously; this sets is_nao_speaking
                                asyncio.create_task(self._speak_on_nao(full_text))
                                self._log_costs(context="model_text_output")

                            self.logger.info("Model finished generation.")
                            self.model_is_speaking = False

                    await asyncio.sleep(0.05)
            finally:
                for task in (display_task, commentary_task):
                    task.cancel()
                for task in (display_task, commentary_task):
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    # -------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------
    def run(self):
        self.logger.info(
            "Starting NAO Gemini Conversation Demo with tools, audio, vision, cost logging, and auto commentary."
        )
        try:
            asyncio.run(self.run_gemini())
        except Exception as e:
            self.logger.error(f"Error: {e}")
        finally:
            cv2.destroyAllWindows()
            self.shutdown()


if __name__ == "__main__":
    NaoGeminiConversation().run()
