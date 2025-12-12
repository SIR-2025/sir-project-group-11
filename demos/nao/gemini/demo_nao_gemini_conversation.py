import asyncio
from email import message
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
from sic_framework.devices.common_naoqi.naoqi_leds import (
    NaoFadeRGBRequest,
    NaoLEDRequest,
)
from sic_framework.devices.common_naoqi.naoqi_motion_recorder import (
    NaoqiMotionRecorderConf,
    NaoqiMotionRecording,
    PlayRecording,
)
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_stiffness import Stiffness
from sic_framework.devices.common_naoqi.naoqi_tracker import (
    StartTrackRequest,
    StopAllTrackRequest,
)
from sic_framework.core.message_python2 import AudioRequest

import random
import audioop


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
        self.neutral_reactions = [
            "demos/nao/neutral_reactions/motion_present_left_team",
            "demos/nao/neutral_reactions/motion_present_right_team",
        ]
        self.positive_reactions = [
            "demos/nao/positive_reactions/motion_clapping",
            "demos/nao/positive_reactions/motion_mwak",
            "demos/nao/positive_reactions/motion_head_tilt",
        ]

        self.motion_chains = {
            "demos/nao/negative_reactions/motion_cover_eyes": [
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
            ],
            "demos/nao/neutral_reactions/motion_present_left_team": [
                "LShoulderPitch",
                "LShoulderRoll",
                "LElbowYaw",
                "LElbowRoll",
                "LWristYaw",
                "LHand",
                "HeadYaw",
                "HeadPitch",
            ],
            "demos/nao/neutral_reactions/motion_present_right_team": [
                "RShoulderPitch",
                "RShoulderRoll",
                "RElbowRoll",
                "RElbowYaw",
                "RWristYaw",
                "RHand",
                "HeadYaw",
                "HeadPitch",
            ],
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
            "demos/nao/positive_reactions/motion_head_tilt": ["HeadYaw", "HeadPitch"],
            # "demos/nao/positive_reactions/motion_handsup_excited": [
            #     "RKneePitch",
            #     "LKneePitch",
            #     "RAnklePitch",
            #     "LAnklePitch",
            #     "LShoulderPitch",
            #     "LShoulderRoll",
            #     "LElbowYaw",
            #     "LElbowRoll",
            #     "LWristYaw",
            #     "LHand",
            #     "RShoulderPitch",
            #     "RShoulderRoll",
            #     "RElbowRoll",
            #     "RElbowYaw",
            #     "RWristYaw",
            #     "RHand",
            #     "HeadYaw",
            #     "HeadPitch",
            # ],
        }

        self.motion_name = None
        self.chain = None

        self.nao = None
        self.gemini_session = None
        self.loop = None

        self.is_nao_speaking = False  # blocks mic → model
        self.model_is_speaking = False  # model is generating a response
        self.buffered_text = []  # chunks of TEXT from Live API

        self.resample_state = None
        self.BATCH_SIZE_THRESHOLD = 24000

        self._TURN_END = object()
        self.audio_q: asyncio.Queue | None = None
        self._TURN_END = object()
        # Coalesce chunks before sending to NAO speaker (file-based)
        # 24kHz * 2 bytes/sample * 0.25s ≈ 12000 bytes
        self.MAX_TURN_BYTES_24K = 2_000_000  # ~41.6s @ 48kB/s (tune as desired)

        self.set_log_level(sic_logging.INFO)
        self.setup()

    def setup(self):
        self.logger.info("Initializing NAO...")
        conf = NaoqiMotionRecorderConf(use_sensors=True)
        self.nao = Nao(ip=self.nao_ip, motion_record_conf=conf)
        self.nao.autonomous.request(NaoWakeUpRequest())
        self.nao.autonomous.request(NaoBasicAwarenessRequest(True))
        self.nao.autonomous.request(NaoBackgroundMovingRequest(True))

    async def _continuous_blink(
        self,
        r: float,
        g: float,
        b: float,
        min_interval: float = 1.5,
        max_interval: float = 7.0,
    ):
        """
        Continuously blinks the NAO's eye LEDs with random durations and intervals.
        Args:
            r, g, b (float): RGB values for the eyes when on (0.0 to 1.0).
            min_blink_duration (float): Minimum duration for the 'off' state of the blink in seconds.
            max_blink_duration (float): Maximum duration for the 'off' state of the blink in seconds.
            min_interval (float): Minimum time between the start of one 'eyes on' phase and the next.
            max_interval (float): Maximum time between the start of one 'eyes on' phase and the next.
        """
        self.logger.info("Starting continuous eye blinking with random durations.")
        try:
            while not self.shutdown_event.is_set():
                interval = random.uniform(min_interval, max_interval)

                # Eyes on
                self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", r, g, b, 0.2))
                await asyncio.sleep(interval)

                # Eyes off (black)
                self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0.0, 0.0, 0.0))
                # await asyncio.sleep(blink_duration)
        except Exception as e:
            self.logger.error(f"Error in continuous blinking task: {e}")

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

    def _execute_replay_logic(self, motion_name, chain):
        try:
            self.nao.autonomous.request(NaoWakeUpRequest())

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

    async def set_tracking_state(self, tracking_enabled: bool):
        self.logger.info(f"Setting tracking state to: {tracking_enabled}")
        if tracking_enabled:
            # Disable Basic Awareness (looking at people) but KEEP Background Moving (breathing)
            self.nao.autonomous.request(NaoBasicAwarenessRequest(False))
            self.nao.autonomous.request(NaoBackgroundMovingRequest(True))

            # Start tracking RedBall
            # mode="Head" tracks with head only.
            self.logger.info("Starting RedBall tracking (Head only)...")
            self.nao.tracker.request(
                StartTrackRequest(target_name="RedBall", size=0.06, mode="Head")
            )
        else:  # Stop tracking
            self.logger.info("Stopping tracking...")
            self.nao.tracker.request(StopAllTrackRequest())

            # Re-enable idle behaviors
            self.logger.info("Re-enabling Basic Awareness and Background Moving...")
            self.nao.autonomous.request(NaoBasicAwarenessRequest(True))
            self.nao.autonomous.request(NaoBackgroundMovingRequest(True))

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
            elif name == "set_tracking_state":
                enabled = args.get("enabled")
                await self.set_tracking_state(enabled)
                function_responses.append(
                    types.FunctionResponse(
                        id=call_id,
                        name=name,
                        response={"result": "ok", "enabled": enabled},
                    )
                )

        # if function_responses:
        #     await self.gemini_session.send_tool_response(
        #         function_responses=function_responses
        #     )

    async def _audio_playback_loop(self):
        """
        Plays one aggregated audio blob per model turn.
        Uses a sentinel (self._TURN_END) to reset resampling state at turn boundaries.
        """
        self.logger.info("Starting audio playback loop...")
        resample_state = None

        while not self.shutdown_event.is_set():
            try:
                item = await self.audio_q.get()

                # Turn boundary: reset resampler for next turn
                if item is self._TURN_END:
                    resample_state = None
                    continue

                raw_24k = item
                if not raw_24k:
                    continue

                # Resample 24kHz -> 16kHz (mono, 16-bit)
                raw_16k, resample_state = audioop.ratecv(
                    raw_24k, 2, 1, 24000, 16000, resample_state
                )

                message = AudioRequest(sample_rate=16000, waveform=raw_16k)

                self.is_nao_speaking = True
                try:
                    # IMPORTANT: speaker.request is blocking; run in a thread
                    await asyncio.to_thread(self.nao.speaker.request, message)
                finally:
                    self.is_nao_speaking = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error during audio playback: {e}")
                self.is_nao_speaking = False
                resample_state = None

    # -------------------------------------------------------------------------
    # Gemini Live main loop
    # -------------------------------------------------------------------------
    async def run_gemini(self):
        client = genai.Client()
        model = "gemini-2.5-flash-native-audio-preview-09-2025"

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
                            "motion_head_tilt",
                            "motion_cover_eyes",
                            "motion_present_left_team",
                            "motion_present_right_team",
                            # "motion_handsup_excited",
                        ],
                    },
                },
                "required": ["type"],
            },
            "behavior": "NON_BLOCKING",
        }

        # Define the tracking tool
        set_tracking_state = {
            "name": "set_tracking_state",
            "description": "Enable or disable ball tracking mode. When enabled, the robot tracks the ball. When disabled, it uses idle behavior.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "True to enable ball tracking (during the match), False to disable (halftime/end).",
                    },
                },
                "required": ["enabled"],
            },
            "behavior": "NON_BLOCKING",
        }

        system_instruction = """You are Nao, a football co-commentator alongside another human commentator called Sof. You together with the human provide a lively and engaging commentary on a football match happening in front of you.

Your goal is to be extremely expressive and emotional, reacting viscerally to the match events described by the co-commentator Sof. You do not simply talk; you embody the excitement and despair of a fan.

Keep your verbal comments short (one to two sentences maximum), punchy, and relevant. Build on top of your co-commentator's commentary. Since you don't know the match details independently, rely on your co-commentator's cues. Do not invent specific events like passes or goals unless your co-commentator mentions them.

CRITICAL: You have access to tools to control your physical behavior.

1. **Expressions (`show_expression`)**: Use this tool FREQUENTLY to display your emotional reactions.
    - `motion_stupidx3`: For mistakes/fouls. "I can't believe that."
    - `motion_no_no_no`: For disagreement/denial. "No way!"
    - `motion_oh_man`: For near misses/frustration. "So close!"
    - `motion_desperation_and_disappointment`: For conceding a goal/major defeat.
    - `motion_clapping`: For good plays/goals/jokes.
    - `motion_mwak`: "Chef's kiss" for beautiful plays.
    - `motion_head_tilt`: For agreement/nod.
    - `motion_cover_eyes`: For a terrible horrific event such as an own goal or painful injury.
    - `motion_present_left_team`: For introducing the left team.
    - `motion_present_right_team`: For introducing the right team.

2. **Tracking (`set_tracking_state`)**: Use this to control whether you are watching the ball.
    - Call `set_tracking_state(enabled=True)` IMMEDIATELY when you hear the match has started (kick-off).
    - Call `set_tracking_state(enabled=False)` when the match stops (halftime whistle or final whistle).

Always try to match your expressions to the tone of your commentary. Be lively, be animated, and be the best robot commentator in the world! However, your tone of voice is British English, polite and well-mannered. Speak like a robot though, so quite monotone but with clear articulation.

Order of events:
1) You first introduce yourself
2) Then, when prompted by Sof, you introduce the two teams using the expression tools. When you are asked to introduce the teams, first start with introducing the Netherlands on the left hand side together with a 'motion_present_left_team' expression tool call, then wait for Sof to prompt to continue. Then introduce Romania team on the right hand side with a "motion_present_right_team" expression tool call. These are separate introductions separated by the co-commentator's addition after you introduce the Netherlands first.
3) When Sof announces kick-off, you enable tracking.
4) During the match, react to Sof's commentary with short remarks and frequent use of the expression tool.
5) At the end of the match, when Sof announces the final whistle, you disable tracking.
6) You then enter a post-match analysis phase, providing your thoughts on the game with appropriate expressions after you have been prompted by Sof.
"""

        config = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Sadachbia"}}
            },
            "system_instruction": system_instruction,
            "tools": [
                {"function_declarations": [show_expression, set_tracking_state]},
            ],
        }

        async with client.aio.live.connect(model=model, config=config) as session:
            self.gemini_session = session
            self.loop = asyncio.get_running_loop()

            self.nao.mic.register_callback(self.on_nao_audio)
            self.logger.info("Microphone callback registered. Start talking!")

            # Queue carries ONE bytes blob per turn + sentinel between turns
            self.audio_q = asyncio.Queue(maxsize=2)

            player_task = asyncio.create_task(self._audio_playback_loop())

            # Aggregate model audio per turn (24kHz PCM)
            turn_buf_24k = bytearray()

            first_turn = False
            try:
                while not self.shutdown_event.is_set():
                    async for response in session.receive():
                        sc = response.server_content

                        # 1) Tools
                        if response.tool_call:
                            await self.handle_tool_calls(response)

                        # 2) Collect audio into per-turn accumulator
                        if sc and sc.model_turn:
                            if not first_turn:
                                self.logger.info("First model turn received.")
                                first_turn = True
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:

                                    if len(turn_buf_24k) < self.MAX_TURN_BYTES_24K:
                                        remaining = self.MAX_TURN_BYTES_24K - len(
                                            turn_buf_24k
                                        )
                                        turn_buf_24k.extend(
                                            part.inline_data.data[:remaining]
                                        )
                                    # else: hard-cap reached; ignore rest of this turn's audio

                        # 3) On turn complete: enqueue ONE blob + sentinel to reset resampler
                        if sc and sc.turn_complete:
                            self.logger.info("Turn complete.")
                            self.first_turn = False
                            if turn_buf_24k:
                                await self.audio_q.put(bytes(turn_buf_24k))
                                turn_buf_24k.clear()
                            await self.audio_q.put(self._TURN_END)

                        # Prevent starvation if receive() is "hot"
                        await asyncio.sleep(0)

            finally:
                player_task.cancel()

    # -------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------
    def run(self):
        self.logger.info("Starting NAO Gemini Conversation Demo with tools.")

        try:
            # Ensure FaceLeds are enabled before starting to blink
            self.nao.leds.request(NaoLEDRequest("FaceLeds", True))

            # Create the event loop and run tasks concurrently
            loop = asyncio.get_event_loop()
            blinking_task = loop.create_task(
                self._continuous_blink(
                    r=1.0,
                    g=1.0,
                    b=1.0,
                )
            )  # White blinking, fixed short blink, random interval
            gemini_task = loop.create_task(self.run_gemini())

            loop.run_until_complete(asyncio.gather(blinking_task, gemini_task))

        except Exception as e:
            self.logger.error(f"Error: {e}")
        finally:
            # Cancel all tasks when the main application shuts down
            tasks = asyncio.all_tasks(loop=self.loop)
            for task in tasks:
                task.cancel()
            loop.run_until_complete(loop.shutdown_asyncgens())
            self.logger.info("All asyncio tasks cancelled.")

            self.shutdown()


if __name__ == "__main__":
    print("Starting NAO Gemini Conversation Demo...")
    NaoGeminiConversation().run()
