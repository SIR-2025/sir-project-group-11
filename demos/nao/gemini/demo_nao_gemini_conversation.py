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

        self.nao_ip = "10.0.0.243"

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
    async def perform_expression(self, type: str | None = None):
        if type:
            self.logger.info(f"NAO expression requested with type: {type}")

        else:
            self.logger.info("NAO dance requested with default style")

        # make it sleep for 5 seconds
        await asyncio.sleep(5)
        self.logger.info("NAO dance routine completed.")
        # TODO: hook this into your real expression behavior.
        # Example pseudo-code (replace with actual SIC calls):
        # self.nao.motion.request(NaoqiMotionRequest(behavior="expression_happy"))
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
                type = args.get("type")
                await self.perform_expression(type)
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id,
                        name=name,
                        response={"result": "ok", "type": type},
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
        client = genai.Client()
        model = "gemini-live-2.5-flash-preview"

        # Define the dance tool for the model
        show_expression = {
            "name": "show_expression",
            "description": (
                "Make the NAO robot show a particular facial expression."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "'Normal', 'Happy', 'Sad', 'Angry', 'Surprised', 'Confused'",
                    },
                },
                "required": ["type"],
            },
            "behavior": "NON_BLOCKING",
        }

        config = {
            "response_modalities": ["TEXT"],
            "system_instruction": "You are Nao, a football co-commentator alongside another human commentator called Marcus. You together with the human provide a lively and engaging commentary on a football match happening in front of you. Keep your comments short and relevant to the current state of the match. Build on top of the human commentator. You do not know details about the match, so you cannot make actions up unless the other commentator has specified so, thus keep all of your comments ambiguous or relating on the previous comment. For example, do not say something about a pass or a goal if the commentator has not indicated this. You also have access to several expressions. These expressions are: Normal, Happy, Sad, Angry, Surprised, Confused. Use these expressions to show your emotional reaction to the events happening in the match. You do this by explicitly calling the show_expression tool. E.g. if the human commentator makes a joke, you can respond with a 'Happy' expression. If something unfortunate happens, you can use the 'Sad' expression. Always try to match your expressions to the tone of your commentary. You responses are always one sentence maximum at a time.",
            "tools": [
                {"function_declarations": [show_expression]},
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