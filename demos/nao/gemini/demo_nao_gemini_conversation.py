import asyncio
from google import genai
from google.genai import types
from sic_framework.devices.nao import NaoqiTextToSpeechRequest

from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging

from sic_framework.devices import Nao
from sic_framework.core.message_python2 import AudioMessage


class NaoGeminiConversation(SICApplication):
    """
    NAO Gemini Live TEXT conversation with tool calling.

    - Streams NAO microphone audio to Gemini Live.
    - Buffers Gemini text output and waits for generation_complete.
    - Sends the full text to NAO TTS.
    - Uses function calling to trigger NAO dance when user asks for it.
    """

    def __init__(self):
        super(NaoGeminiConversation, self).__init__()

        self.nao_ip = "10.0.0.242"

        self.nao = None
        self.gemini_session = None
        self.loop = None

        self.is_nao_speaking = False  # blocks mic → model
        self.model_is_speaking = False  # model is generating a response
        self.buffered_text = []  # chunks of TEXT from Live API

        self.set_log_level(sic_logging.INFO)
        self.setup()

    def setup(self):
        self.logger.info("Initializing NAO...")
        self.nao = Nao(ip=self.nao_ip)

    # -------------------------------------------------------------------------
    # NAO-side actions
    # -------------------------------------------------------------------------
    async def perform_nao_dance(self, style: str | None = None):
        """
        Run a dance routine on NAO.

        Replace the body of this method with your real SIC/NAO motion code.
        For example, start a pre-programmed behavior or a motion sequence.
        """
        if style:
            self.logger.info(f"NAO dance requested with style: {style}")

        else:
            self.logger.info("NAO dance requested with default style")

        # make it sleep for 5 seconds
        await asyncio.sleep(5)
        self.logger.info("NAO dance routine completed.")
        # TODO: hook this into your real dance behavior.
        # Example pseudo-code (replace with actual SIC calls):
        # self.nao.motion.request(NaoqiMotionRequest(behavior="dance_basic"))
        # or self.nao.motion_streamer.request(...)
        pass

    # -------------------------------------------------------------------------
    # Audio capture from NAO → Gemini
    # -------------------------------------------------------------------------
    def on_nao_audio(self, message: AudioMessage):
        """
        Mic callback — only forward audio when NAO is not speaking
        and the model is not currently generating output.
        """
        if self.is_nao_speaking or self.model_is_speaking:
            return

        if self.gemini_session and self.loop and not self.loop.is_closed():
            coro = self.gemini_session.send_realtime_input(
                audio=types.Blob(
                    data=message.waveform,
                    mime_type="audio/pcm;rate=16000",
                )
            )
            asyncio.run_coroutine_threadsafe(coro, self.loop)

    # -------------------------------------------------------------------------
    # Tool-call handling
    # -------------------------------------------------------------------------
    async def handle_tool_calls(self, response):
        """
        Handle tool calls emitted by the model.

        Expects response.tool_call.function_calls to contain one or more
        tool invocations. Executes the appropriate action and returns
        FunctionResponse objects back to Gemini.
        """
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
        client = genai.Client(api_key='AIzaSyByaVJkD9c4dxd-7Rb81hucbP-fTZ8HbFA')
        model = "gemini-live-2.5-flash-preview"

        # Define the dance tool for the model
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
                "You are Nao, a friendly robot assistant. "
                "Use clear punctuation. "
                "When the user asks you to dance or refers to you doing a dance, "
                "call the 'start_dance' tool instead of just answering in text."
            ),
            "tools": [
                {"function_declarations": [start_dance_tool]},
            ],
        }

        async with client.aio.live.connect(model=model, config=config) as session:
            self.gemini_session = session
            self.loop = asyncio.get_running_loop()

            # Start NAO microphone streaming
            self.nao.mic.register_callback(self.on_nao_audio)
            self.logger.info("Microphone callback registered. Start talking!")

            while not self.shutdown_event.is_set():
                async for response in session.receive():
                    sc = response.server_content

                    # Handle tool calls (if any)
                    if response.tool_call:
                        await self.handle_tool_calls(response)

                    # ------------------------------------------------------
                    # 1. MODEL STARTS SPEAKING (first chunk of TEXT)
                    # ------------------------------------------------------
                    if response.text is not None:
                        if not self.model_is_speaking:
                            self.model_is_speaking = True
                            self.is_nao_speaking = True
                            self.logger.info("Model started responding; mic muted.")

                        self.buffered_text.append(response.text)

                    # ------------------------------------------------------
                    # 2. MODEL FINISHES ITS TURN (use generation_complete)
                    # ------------------------------------------------------
                    if sc and sc.generation_complete:
                        full_text = "".join(self.buffered_text).strip()
                        self.buffered_text = []

                        if full_text:
                            self.logger.info(f"Full model response: {full_text}")
                            # Speak on NAO
                            self.nao.tts.request(
                                NaoqiTextToSpeechRequest(full_text),
                                block=True,
                            )

                        self.logger.info("NAO finished speaking; mic unmuted.")
                        self.model_is_speaking = False
                        self.is_nao_speaking = False

                await asyncio.sleep(0.05)

    # -------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------
    def run(self):
        self.logger.info("Starting NAO Gemini Conversation Demo with tools.")

        try:
            asyncio.run(self.run_gemini())
        except Exception as e:
            self.logger.error(f"Error: {e}")
        finally:
            self.shutdown()


if __name__ == "__main__":
    NaoGeminiConversation().run()