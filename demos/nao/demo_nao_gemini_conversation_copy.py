import asyncio
from google import genai
from google.genai import types
from sic_framework.devices.nao import NaoqiTextToSpeechRequest, NaoLEDRequest

from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging

from sic_framework.devices import Nao
from sic_framework.core.message_python2 import AudioMessage

from sic_framework.devices.common_naoqi.naoqi_motion import (
    NaoPostureRequest,
    NaoqiAnimationRequest,
)


class NaoGeminiConversation(SICApplication):
    """
    NAO Gemini Live TEXT conversation with tool calling.

    - Streams NAO microphone audio to Gemini Live.
    - Buffers Gemini text output and waits for generation_complete.
    - Sends the full text to NAO TTS.
    - Uses function calling to trigger NAO dance when user asks for it.
    - Uses function calling to change LED colors based on emotional state.
    """

    # Extended emotion-to-color mapping (RGB hex values)
    LED_COLORS = {
        "happy": 0x00FF00,        # Green
        "sad": 0x0000FF,          # Blue
        "frustrated": 0xFF0000,   # Red
        "neutral": 0xFFFFFF,      # White
        "excited": 0xFFFF00,      # Yellow
        "curious": 0x8000FF,      # Purple
        "thinking": 0x00FFFF,     # Cyan
        "confident": 0x00AA00,    # Dark Green
        "uncertain": 0xFF8000,    # Orange
        "apologetic": 0xFF00FF,   # Magenta
    }

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
    def set_led_color(self, emotion: str):
        """
        Set NAO's eye and chest LED colors based on emotion.

        Args:
            emotion: The emotional state (must be in LED_COLORS dict)
        """
        if emotion not in self.LED_COLORS:
            self.logger.warning(f"Unknown emotion: {emotion}. Using neutral.")
            emotion = "neutral"

        color = self.LED_COLORS[emotion]
        self.logger.info(f"Setting LEDs to {emotion} (color: {hex(color)})")

        # Set eye LEDs
        self.nao.leds.request(
            NaoLEDRequest("FaceLeds", color, block=False)
        )

        # Set chest LED
        self.nao.leds.request(
            NaoLEDRequest("ChestLeds", color, block=False)
        )

    # Individual LED color methods for each emotion (for tool calls)
    async def set_led_happy(self):
        """Set LEDs to happy (green)."""
        self.set_led_color("happy")

    async def set_led_sad(self):
        """Set LEDs to sad (blue)."""
        self.set_led_color("sad")

    async def set_led_frustrated(self):
        """Set LEDs to frustrated (red)."""
        self.set_led_color("frustrated")

    async def set_led_neutral(self):
        """Set LEDs to neutral (white)."""
        self.set_led_color("neutral")

    async def set_led_excited(self):
        """Set LEDs to excited (yellow)."""
        self.set_led_color("excited")

    async def set_led_curious(self):
        """Set LEDs to curious (purple)."""
        self.set_led_color("curious")

    async def set_led_thinking(self):
        """Set LEDs to thinking (cyan)."""
        self.set_led_color("thinking")

    async def set_led_confident(self):
        """Set LEDs to confident (dark green)."""
        self.set_led_color("confident")

    async def set_led_uncertain(self):
        """Set LEDs to uncertain (orange)."""
        self.set_led_color("uncertain")

    async def set_led_apologetic(self):
        """Set LEDs to apologetic (magenta)."""
        self.set_led_color("apologetic")

    async def perform_nao_dance(self, style: str | None = None):
        """ Run a NAO motion routine instead of just sleeping. 
        You can customize which animation to run based on 'style'."""

        try:
            self.logger.info(f"Executing NAO motion (style={style})")

            # Always stand first (safe starting posture)
            self.nao.motion.request(NaoPostureRequest("Stand", 0.5))
            await asyncio.sleep(1)

            # Select animation based on style
            if style == "hey" or style is None:
                animation = "animations/Stand/Gestures/Hey_1"
            elif style == "joy":
                animation = "animations/Stand/Gestures/Happy_1"
            elif style == "bow":
                animation = "animations/Stand/Gestures/BowShort_1"
            else:
                # Default fallback
                animation = "animations/Stand/Gestures/Hey_1"

            self.logger.info(f"Playing NAO animation: {animation}")
            self.nao.motion.request(NaoqiAnimationRequest(animation))

            # Optional: LED reset afterwards
            await asyncio.sleep(1)
            self.nao.leds.request(NaoLEDRequest("FaceLeds", True))

            self.logger.info("Motion routine completed.")

        except Exception as e:
            self.logger.error(f"Error running NAO motion: {e}")

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

            # LED emotion tools
            elif name == "set_led_happy":
                await self.set_led_happy()
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id, name=name, response={"result": "ok", "emotion": "happy"}
                    )
                )

            elif name == "set_led_sad":
                await self.set_led_sad()
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id, name=name, response={"result": "ok", "emotion": "sad"}
                    )
                )

            elif name == "set_led_frustrated":
                await self.set_led_frustrated()
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id, name=name, response={"result": "ok", "emotion": "frustrated"}
                    )
                )

            elif name == "set_led_neutral":
                await self.set_led_neutral()
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id, name=name, response={"result": "ok", "emotion": "neutral"}
                    )
                )

            elif name == "set_led_excited":
                await self.set_led_excited()
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id, name=name, response={"result": "ok", "emotion": "excited"}
                    )
                )

            elif name == "set_led_curious":
                await self.set_led_curious()
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id, name=name, response={"result": "ok", "emotion": "curious"}
                    )
                )

            elif name == "set_led_thinking":
                await self.set_led_thinking()
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id, name=name, response={"result": "ok", "emotion": "thinking"}
                    )
                )

            elif name == "set_led_confident":
                await self.set_led_confident()
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id, name=name, response={"result": "ok", "emotion": "confident"}
                    )
                )

            elif name == "set_led_uncertain":
                await self.set_led_uncertain()
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id, name=name, response={"result": "ok", "emotion": "uncertain"}
                    )
                )

            elif name == "set_led_apologetic":
                await self.set_led_apologetic()
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id, name=name, response={"result": "ok", "emotion": "apologetic"}
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
        client = genai.Client(api_key="AIzaSyAJUMcSkM5WSl8yOflrr6gODe-p68aNe2s")
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

        # Define LED emotion tools
        led_tools = [
            {
                "name": "set_led_happy",
                "description": "Set NAO's eye and chest LEDs to green (happy emotion). Use when expressing joy, happiness, or positive emotions.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "set_led_sad",
                "description": "Set NAO's eye and chest LEDs to blue (sad emotion). Use when expressing sadness, disappointment, or melancholy.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "set_led_frustrated",
                "description": "Set NAO's eye and chest LEDs to red (frustrated/angry emotion). Use when expressing frustration, anger, or irritation.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "set_led_neutral",
                "description": "Set NAO's eye and chest LEDs to white (neutral emotion). Use for neutral, calm, or default states.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "set_led_excited",
                "description": "Set NAO's eye and chest LEDs to yellow (excited emotion). Use when expressing excitement, enthusiasm, or high energy.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "set_led_curious",
                "description": "Set NAO's eye and chest LEDs to purple (curious emotion). Use when expressing curiosity, interest, or wonder.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "set_led_thinking",
                "description": "Set NAO's eye and chest LEDs to cyan (thinking emotion). Use when processing information, thinking, or contemplating.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "set_led_confident",
                "description": "Set NAO's eye and chest LEDs to dark green (confident emotion). Use when expressing confidence, assurance, or certainty.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "set_led_uncertain",
                "description": "Set NAO's eye and chest LEDs to orange (uncertain emotion). Use when expressing uncertainty, confusion, or doubt.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "set_led_apologetic",
                "description": "Set NAO's eye and chest LEDs to magenta (apologetic emotion). Use when apologizing or expressing regret.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ]

        config = {
            "response_modalities": ["TEXT"],
            "system_instruction": (
                "You are Nao, a friendly robot assistant. "
                "Use clear punctuation. "
                "When the user asks you to dance or refers to you doing a dance, "
                "call the 'start_dance' tool instead of just answering in text. "
                "\n\n"
                "IMPORTANT: You have LED lights on your eyes and chest that can display emotions. "
                "You MUST call the appropriate LED emotion tool (set_led_*) at the START of your response "
                "to match the emotional tone of what you're about to say. "
                "Choose from: happy (green), sad (blue), frustrated (red), neutral (white), "
                "excited (yellow), curious (purple), thinking (cyan), confident (dark green), "
                "uncertain (orange), or apologetic (magenta). "
                "\n\n"
                "For example:\n"
                "- If answering a question confidently, call set_led_confident first\n"
                "- If expressing frustration or saying you can't do something, call set_led_frustrated\n"
                "- If expressing curiosity about something the user said, call set_led_curious\n"
                "- If thinking or processing, call set_led_thinking\n"
                "- Default to set_led_neutral for routine responses\n"
                "\n"
                "Always set your LEDs to match your emotional tone before or during your verbal response."
            ),
            "tools": [
                {"function_declarations": [start_dance_tool] + led_tools},
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
