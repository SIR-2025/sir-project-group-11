import asyncio
import random
import audioop
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.core.message_python2 import AudioMessage
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import (
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

# Optional ASR dependency: faster-whisper
#   pip install faster-whisper
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


@dataclass
class VADConfig:
    sample_rate: int = 16000
    energy_threshold: int = 450  # tune for NAO mic (RMS)
    min_speech_ms: int = 180  # ignore tiny spikes
    end_silence_ms: int = 650  # silence needed to close the turn
    max_utterance_ms: int = 12000  # safety cap
    pre_roll_ms: int = 200  # keep a tiny lead-in


class SimpleEnergyVAD:
    """
    Energy-based VAD with end-of-speech detection.
    - Feed 16kHz PCM16LE mono frames.
    - Returns:
        "speech" -> keep buffering
        "end"    -> utterance ended (emit buffered audio)
        None     -> ignore (silence / not started)
    """

    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self._in_speech = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._utterance_ms = 0.0
        self._pre_roll = bytearray()
        self._buf = bytearray()

    def _frame_ms(self, pcm16: bytes) -> float:
        return (len(pcm16) / 2) / self.cfg.sample_rate * 1000.0

    def process(self, pcm16: bytes) -> Optional[str]:
        if not pcm16:
            return None

        ms = self._frame_ms(pcm16)
        self._utterance_ms += ms

        # Maintain pre-roll (always)
        self._pre_roll.extend(pcm16)
        pre_roll_bytes = int(self.cfg.pre_roll_ms / 1000.0 * self.cfg.sample_rate * 2)
        if len(self._pre_roll) > pre_roll_bytes:
            self._pre_roll = self._pre_roll[-pre_roll_bytes:]

        energy = audioop.rms(pcm16, 2)

        if energy >= self.cfg.energy_threshold:
            # Speech detected
            if not self._in_speech:
                self._in_speech = True
                self._buf.extend(self._pre_roll)
                self._pre_roll.clear()

            self._buf.extend(pcm16)
            self._speech_ms += ms
            self._silence_ms = 0.0
            return "speech"

        # Silence
        if not self._in_speech:
            return None

        self._silence_ms += ms
        if self._utterance_ms >= self.cfg.max_utterance_ms:
            return "end"

        if (
            self._speech_ms >= self.cfg.min_speech_ms
            and self._silence_ms >= self.cfg.end_silence_ms
        ):
            return "end"

        # Still waiting (in-speech but not enough trailing silence yet)
        return "speech"

    def pop_audio(self) -> bytes:
        audio = bytes(self._buf)
        self.reset()
        return audio


class NaoGeminiConversation(SICApplication):
    """
    NAO live mic -> local VAD -> local ASR -> fast text LLM -> NAO TTS
    - Uses function calling for expressions + tracking.
    - Uses Gemini text model (fast text) instead of audio-only Live model.
    """

    def __init__(self):
        super(NaoGeminiConversation, self).__init__()

        self.nao_ip = "10.0.0.245"

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
        }

        self.nao: Optional[Nao] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self.is_nao_speaking = False
        self.model_is_busy = False

        self.emotion = None

        self.pending_expression = None

        self.vad = SimpleEnergyVAD(VADConfig(sample_rate=16000))
        self._vad_lock = asyncio.Lock()

        self.client = genai.Client(api_key="AIzaSyAvfE2eOPfvrLLQqiltFi-ne3s0XlltTQs")
        self.model = "gemini-2.5-flash"  # fast text model

        # Chat/session state for the text model
        self.history = []

        self.MIN_TRANSCRIPT_CHARS = 8
        self.MIN_TRANSCRIPT_WORDS = 2

        # Local ASR
        self.asr = None
        if WhisperModel is not None:
            # pick a small model for latency; adjust to "tiny", "base", "small" etc.
            # device="cpu" is usually fine; for GPU add device="cuda"
            self.asr = WhisperModel("small.en", device="cpu", compute_type="int8")

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
        self.logger.info("Starting continuous eye blinking.")
        try:
            while not self.shutdown_event.is_set():
                interval = random.uniform(min_interval, max_interval)
                self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", r, g, b, 0.2))
                await asyncio.sleep(interval)
                self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0.0, 0.0, 0.0))
        except Exception as e:
            self.logger.error(f"Blinking task error: {e}")

    async def set_tracking_state(self, tracking_enabled: bool):
        self.logger.info(f"Setting tracking state to: {tracking_enabled}")
        if tracking_enabled:
            self.nao.autonomous.request(NaoBasicAwarenessRequest(False))
            self.nao.autonomous.request(NaoBackgroundMovingRequest(True))
            self.logger.info("Starting RedBall tracking (Head only)...")
            self.nao.tracker.request(
                StartTrackRequest(target_name="RedBall", size=0.06, mode="Head")
            )
        else:
            self.logger.info("Stopping tracking...")
            self.nao.tracker.request(StopAllTrackRequest())
            self.logger.info("Re-enabling idle behaviors...")
            self.nao.autonomous.request(NaoBasicAwarenessRequest(True))
            self.nao.autonomous.request(NaoBackgroundMovingRequest(True))

    async def perform_expression(self, type: str | None = None):
        self.logger.info(f"NAO expression requested: {type}")
        if not type or type == "Normal":
            return

        full_path = next(
            (k for k in self.motion_chains if k.endswith(f"/{type}")), None
        )
        if not full_path:
            self.logger.error(f"Motion {type} not found.")
            return
        
        if "positive" in full_path: 
            self.emotion="Happy"
        elif "negative" in full_path:
            self.emotion="Angry"
        else:
            self.emotion="Calm"

        self.display_emotion()

        chain = self.motion_chains[full_path]
        asyncio.create_task(
            asyncio.to_thread(self._execute_replay_logic, full_path, chain)
        )

        self.emotion="Calm"
        self.display_emotion()

    def _execute_replay_logic(self, motion_name, chain):
        try:
            self.nao.autonomous.request(NaoWakeUpRequest())
            self.nao.stiffness.request(Stiffness(stiffness=0.7, joints=chain))
            recording = NaoqiMotionRecording.load(motion_name)
            self.nao.motion_record.request(PlayRecording(recording))
            self.nao.motion.request(NaoPostureRequest("Stand", 0.5), block=False)
        except Exception as e:
            self.logger.error(f"Error replaying motion: {e}")

    def display_emotion(self):
        """
        Helper function to announce an emotion and set the LEDs.
        """
        if self.shutdown_event.is_set():
            return
        
        emotions = {
            "Happy": (0.0, 1.0, 0.0, "I am feeling happy!"),        # Green
            "Angry": (1.0, 0.0, 0.0, "I am feeling angry!"),        # Red
            "Calm": (1.0, 1.0, 1.0, "I am feeling calm."),          # White
        }

        self.logger.info(f"Displaying emotion: {self.emotion}")

        r, g, b, _ = emotions[self.emotion]
        self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", r, g, b, 0.5))
        self.nao.leds.request(NaoFadeRGBRequest("ChestLeds", r, g, b, 0.5))

    async def _nao_say(self, text: str):
        if not text.strip():
            return
        self.is_nao_speaking = True
        try:
            # Blocking NAO TTS call -> run in thread
            await asyncio.to_thread(
                self.nao.tts.request, NaoqiTextToSpeechRequest(text)
            )
        finally:
            self.is_nao_speaking = False

    def on_nao_audio(self, message: AudioMessage):
        """
        Mic callback:
        - gates while NAO is speaking or model busy
        - runs VAD incrementally
        - on end-of-speech, schedules a turn for ASR + LLM
        """
        if self.is_nao_speaking or self.model_is_busy:
            return

        if not self.loop or self.loop.is_closed():
            return

        # VAD processing is cheap; keep it in callback
        state = self.vad.process(message.waveform)
        if state == "end":
            audio = self.vad.pop_audio()
            self.logger.info(
                f"[VAD] Turn ended. Buffered audio size = {len(audio)} bytes"
            )
            asyncio.run_coroutine_threadsafe(self.process_user_turn(audio), self.loop)

    async def transcribe(self, pcm16: bytes) -> str:
        if self.asr is None:
            raise RuntimeError(
                "ASR not available. Install faster-whisper (pip install faster-whisper)."
            )

        # faster-whisper expects a file path or numpy array; easiest is temp WAV.
        # To avoid extra deps, write raw PCM into a WAV container via standard library.
        import wave
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            with wave.open(f, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm16)
            f.flush()

            segments, _info = self.asr.transcribe(
                f.name, language=None, vad_filter=False
            )
            text_parts = [seg.text for seg in segments]
            return " ".join(t.strip() for t in text_parts if t.strip()).strip()

    async def handle_tool_calls_from_response(self, function_calls):
        """
        Executes tool calls and returns a list of FunctionResponse to feed back to the model.
        """
        responses: list[types.FunctionResponse] = []

        for fc in function_calls:
            name = fc.name
            args = dict(fc.args) if fc.args is not None else {}
            call_id = fc.id

            self.logger.info(f"Tool call: {name} args={args} id={call_id}")

            if name == "show_expression":
                motion_type = args.get("type")
                self.pending_expression = motion_type
                responses.append(
                    types.FunctionResponse(
                        id=call_id,
                        name=name,
                        response={"result": "ok", "type": motion_type},
                    )
                )
            elif name == "set_tracking_state":
                enabled = bool(args.get("enabled"))
                await self.set_tracking_state(enabled)
                responses.append(
                    types.FunctionResponse(
                        id=call_id,
                        name=name,
                        response={"result": "ok", "enabled": enabled},
                    )
                )
            else:
                responses.append(
                    types.FunctionResponse(
                        id=call_id,
                        name=name,
                        response={"result": "unhandled"},
                    )
                )

        return responses

    async def process_user_turn(self, pcm16: bytes):
        """
        VAD-delimited user audio -> ASR -> Gemini text -> NAO TTS.
        Handles function calling loop if the model requests tools.
        """
        if self.model_is_busy:
            return

        self.model_is_busy = True
        try:
            self.logger.info("Processing user turn...")
            user_text = await self.transcribe(pcm16)
            user_text = (user_text or "").strip()
            if not user_text:
                self.logger.info("[ASR] Empty transcript; skipping.")
                return

            # Skip very short / low-signal transcripts
            if (
                len(user_text) < self.MIN_TRANSCRIPT_CHARS
                or len(user_text.split()) < self.MIN_TRANSCRIPT_WORDS
            ):
                self.logger.info(
                    f"[ASR] Transcript too short; skipping Gemini. Text='{user_text}'"
                )
                return

            self.logger.info(f"ASR: {user_text}")

            # ----- Tools (function declarations) -----
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
                            ],
                        },
                    },
                    "required": ["type"],
                },
            }

            set_tracking_state = {
                "name": "set_tracking_state",
                "description": "Enable or disable ball tracking mode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "description": "True to enable ball tracking, False to disable.",
                        },
                    },
                    "required": ["enabled"],
                },
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

            Never call the expression tool multiple times in a row without any verbal commentary in between. Always say something before using the expression tool again.
            
            Do NOT hallucinate details. You are only allowed to talk about what your co-commentator Sof says. If Sof does not mention any events, do not invent them yourself. Just give a general comment. If the input is unclear, ask for clarification."""

            config = {
                "system_instruction": system_instruction,
                "tools": [
                    {"function_declarations": [show_expression, set_tracking_state]}
                ],
            }

            # ----- Build contents with history -----
            contents = []
            contents.extend(self.history)
            contents.append({"role": "user", "parts": [{"text": user_text}]})

            # ----- Function-calling loop -----
            final_text = ""
            max_tool_roundtrips = 3

            for _ in range(max_tool_roundtrips):
                self.logger.info("Sending contents to Gemini model...")
                resp = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                self.logger.info("Received response from Gemini model: ", resp)
                # Capture model text (may be empty if it only emits tool calls)
                if getattr(resp, "text", None):
                    self.logger.info(f"[GEMINI] Text: {resp.text}")
                    final_text = resp.text

                function_calls = getattr(resp, "function_calls", None) or []
                if not function_calls:
                    break

                tool_responses = await self.handle_tool_calls_from_response(
                    function_calls
                )

                # Append model function call + tool responses back into the conversation
                contents.append(
                    {
                        "role": "model",
                        "parts": [{"function_call": fc} for fc in function_calls],
                    }
                )
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"function_response": fr} for fr in tool_responses],
                    }
                )

            # Persist trimmed history (basic cap)
            self.history.append({"role": "user", "parts": [{"text": user_text}]})
            if final_text:
                self.history.append({"role": "model", "parts": [{"text": final_text}]})

            # Cap history and ensure it never starts with a model turn
            self.history = self.history[-12:]
            if self.history and self.history[0].get("role") == "model":
                self.history = self.history[1:]

            if self.pending_expression:
                await self.perform_expression(self.pending_expression)
                self.pending_expression = None

            if final_text:
                await self._nao_say(final_text)

        except Exception as e:
            self.logger.error(f"process_user_turn error: {e}")
        finally:
            self.model_is_busy = False

    async def main_loop(self):
        self.loop = asyncio.get_running_loop()
        self.nao.mic.register_callback(self.on_nao_audio)
        self.logger.info("Mic callback registered (VAD+ASR+Gemini TEXT).")

        while not self.shutdown_event.is_set():
            await asyncio.sleep(0.1)

    def run(self):
        self.logger.info("Starting NAO Conversation (VAD + ASR + Gemini fast TEXT).")
        try:
            self.nao.leds.request(NaoLEDRequest("FaceLeds", True))

            loop = asyncio.get_event_loop()
            blinking_task = loop.create_task(
                self._continuous_blink(r=1.0, g=1.0, b=1.0)
            )
            main_task = loop.create_task(self.main_loop())

            loop.run_until_complete(asyncio.gather(blinking_task, main_task))

        except Exception as e:
            self.logger.error(f"Run error: {e}")
        finally:
            try:
                tasks = asyncio.all_tasks(loop=asyncio.get_event_loop())
                for t in tasks:
                    t.cancel()
            except Exception:
                pass
            self.shutdown()


if __name__ == "__main__":
    print("Starting NAO Conversation (VAD + ASR + Gemini fast TEXT)...")
    NaoGeminiConversation().run()
