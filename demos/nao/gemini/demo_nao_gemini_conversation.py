import asyncio
from google import genai
from google.genai import types
from sic_framework.devices.nao import NaoqiTextToSpeechRequest

from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging

from sic_framework.devices import Nao
from sic_framework.core.message_python2 import AudioMessage
from sic_framework.devices.common_naoqi.naoqi_autonomous import (
    NaoRestRequest,
    NaoWakeUpRequest,
    NaoBasicAwarenessRequest,
    NaoBackgroundMovingRequest,
)
from sic_framework.devices.common_naoqi.naoqi_motion_recorder import (
    NaoqiMotionRecorderConf,
    NaoqiMotionRecording,
    PlayRecording,
)
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_stiffness import Stiffness
import random


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

        self.nao_ip = "10.0.0.245"

        self.negative_reactions = [
            "demos/nao/negative_reactions/motion_stupidx3",
            "demos/nao/negative_reactions/motion_no_no_no",
            "demos/nao/negative_reactions/motion_oh_man",
            "demos/nao/negative_reactions/motion_desperation_and_disappointment",
        ]
        self.positive_reactions = [
            "demos/nao/positive_reactions/motion_clapping",
            "demos/nao/positive_reactions/motion_mwak",
            "demos/nao/positive_reactions/motion_yay",
            "demos/nao/positive_reactions/motion_yeah",
        ]

        self.motion_chains = {
            "demos/nao/negative_reactions/motion_stupidx3": [
                "LShoulderPitch",
                "LShoulderRoll",
                "LElbowYaw",
                "LElbowRoll",
                "LWristYaw",
                "LHand",
                "RShoulderPitch",
                "RShoulderRoll",
                "RElbowYaw",
                "RElbowRoll",
                "RWristYaw",
                "RHand",
                "HeadYaw",
                "HeadPitch",
            ],
            "demos/nao/negative_reactions/motion_no_no_no": ["HeadYaw", "HeadPitch"],
            "demos/nao/negative_reactions/motion_oh_man": [
                "LShoulderPitch",
                "LShoulderRoll",
                "LElbowYaw",
                "LElbowRoll",
                "LWristYaw",
                "LHand",
                "RShoulderPitch",
                "RShoulderRoll",
                "RElbowYaw",
                "RElbowRoll",
                "RWristYaw",
                "RHand",
                "HeadYaw",
                "HeadPitch",
            ],
            "demos/nao/negative_reactions/motion_desperation_and_disappointment": [
                "LShoulderPitch",
                "LShoulderRoll",
                "LElbowYaw",
                "LElbowRoll",
                "LWristYaw",
                "LHand",
                "RShoulderPitch",
                "RShoulderRoll",
                "RElbowYaw",
                "RElbowRoll",
                "RWristYaw",
                "RHand",
                "HeadYaw",
                "HeadPitch",
                "RKneePitch",
                "LKneePitch",
                "RAnklePitch",
                "LAnklePitch",
                "RAnkleRoll",
                "LAnkleRoll",
                "RHipYawPitch",
                "LHipYawPitch",
                "RHipRoll",
                "LHipRoll",
                "RHipPitch",
                "LHipPitch",
            ],
            "demos/nao/positive_reactions/motion_clapping": [
                "LShoulderPitch",
                "LShoulderRoll",
                "LElbowYaw",
                "LElbowRoll",
                "LWristYaw",
                "LHand",
                "RShoulderPitch",
                "RShoulderRoll",
                "RElbowYaw",
                "RElbowRoll",
                "RWristYaw",
                "RHand",
                "HeadYaw",
                "HeadPitch",
            ],
            "demos/nao/positive_reactions/motion_mwak": [
                "LShoulderPitch",
                "LShoulderRoll",
                "LElbowYaw",
                "LElbowRoll",
                "LWristYaw",
                "LHand",
                "RShoulderPitch",
                "RShoulderRoll",
                "RElbowYaw",
                "RElbowRoll",
                "RWristYaw",
                "RHand",
                "HeadYaw",
                "HeadPitch",
            ],
            "demos/nao/positive_reactions/motion_yay": [
                "RKneePitch",
                "LKneePitch",
                "RAnklePitch",
                "LAnklePitch",
                "LShoulderPitch",
                "LShoulderRoll",
                "LElbowYaw",
                "LElbowRoll",
                "LWristYaw",
                "LHand",
                "RShoulderPitch",
                "RShoulderRoll",
                "RElbowRoll",
                "RElbowYaw",
                "RWristYaw",
                "RHand",
                "HeadYaw",
                "HeadPitch",
            ],
            "demos/nao/positive_reactions/motion_yeah": ["HeadYaw", "HeadPitch"],
        }

        self.motion_name = None
        self.chain = None

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
        conf = NaoqiMotionRecorderConf(use_sensors=True)
        self.nao = Nao(ip=self.nao_ip, motion_record_conf=conf)
        self.nao.autonomous.request(NaoWakeUpRequest())
        self.nao.autonomous.request(NaoBasicAwarenessRequest(True))
        self.nao.autonomous.request(NaoBackgroundMovingRequest(True))

    # -------------------------------------------------------------------------
    # NAO-side actions
    # -------------------------------------------------------------------------
    async def perform_expression(self, type: str | None = None):
        self.logger.info(f"NAO expression requested with type: {type}")

        if not type or type == "Normal":
            self.logger.info("Standard expression requested (no motion).")
            return

        # Find the full path key in motion_chains that ends with the requested type
        full_path = next(
            (key for key in self.motion_chains if key.endswith(f"/{type}")), None
        )

        if full_path:
            self.motion_name = full_path
            self.chain = self.motion_chains[full_path]
            self.logger.info(f"Executing specific motion: {self.motion_name}...")
            asyncio.create_task(
                asyncio.to_thread(
                    self._execute_replay_logic, self.motion_name, self.chain
                )
            )
        else:
            self.logger.error(f"Motion {type} not found in motion chains!")

        self.logger.info("NAO expression routine completed.")

    def _execute_replay_logic(self, motion_name, chain):
        try:
            self.nao.autonomous.request(NaoWakeUpRequest())
            self.logger.info("Replaying action (Stiffness -> Load -> Play)")

            self.nao.stiffness.request(Stiffness(stiffness=0.7, joints=chain))

            recording = NaoqiMotionRecording.load(motion_name)
            self.nao.motion_record.request(PlayRecording(recording))
            self.nao.motion.request(NaoPostureRequest("Stand", 0.5), block=False)
            # self.nao.autonomous.request(NaoRestRequest())
        except Exception as e:
            self.logger.error(f"Error replaying motion: {e}")

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

            if name == "show_expression":
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
            "description": "Trigger a physical motion/expression on the NAO robot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "The ID of the motion to perform.",
                        "enum": [
                            "motion_stupidx3",
                            "motion_no_no_no",
                            "motion_oh_man",
                            "motion_desperation_and_disappointment",
                            "motion_clapping",
                            "motion_mwak",
                            "motion_yay",
                            "motion_yeah",
                        ],
                    },
                },
                "required": ["type"],
            },
            "behavior": "NON_BLOCKING",
        }

        system_instruction = """You are Nao, a football co-commentator alongside another human commentator called Marcus. You together with the human provide a lively and engaging commentary on a football match happening in front of you.

Your goal is to be extremely expressive and emotional, reacting viscerally to the match events described by Marcus. You do not simply talk; you embody the excitement and despair of a fan.

Keep your verbal comments short (one sentence maximum), punchy, and relevant. Build on top of Marcus's commentary. Since you don't know the match details independently, rely on Marcus's cues. Do not invent specific events like passes or goals unless Marcus mentions them.

CRITICAL: You MUST use the `show_expression` tool frequently to display your physical reactions. Do not just say you are excited or disappointed; SHOW it.

Here is a guide on when to use each available expression:

- `motion_stupidx3`: Use this when a player makes a bafflingly stupid mistake, a foul is committed, or the referee makes a terrible call. It signifies "I can't believe how dumb that was."
- `motion_no_no_no`: Use this to express strong disagreement, disbelief at a missed opportunity, or denial that a goal was conceded. "No way, that didn't just happen."
- `motion_oh_man`: Use this for "so close!" moments, near misses, or general frustration. "Oh man, that was unlucky."
- `motion_desperation_and_disappointment`: Use this for major setbacks, conceding a goal, or when the team is playing terribly. It is a full-body slump of defeat.
- `motion_clapping`: Use this to applaud a good pass, a nice save, a goal, or a funny joke by Marcus.
- `motion_mwak`: This is a "chef's kiss" or blowing a kiss. Use it for a beautiful play, a perfect shot, or sarcastically when something is "beautifully terrible."
- `motion_yay`: Use this for high excitement, celebrations, goals, or winning moments. Hands go up in victory!
- `motion_yeah`: A subtle nod or fist pump. Use it for agreement with Marcus, confirmation of a good point, or mild satisfaction.

Always try to match your expressions to the tone of your commentary. Be lively, be animated, and be the best robot commentator in the world!"""

        config = {
            "response_modalities": ["TEXT"],
            "system_instruction": system_instruction,
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
