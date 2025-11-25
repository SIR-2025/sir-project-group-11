import asyncio
from google import genai
from google.genai import types

from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.core.message_python2 import AudioMessage

from sic_framework.devices.common_naoqi.naoqi_motion_recorder import (
    NaoqiMotionRecorderConf,
    NaoqiMotionRecording,
    PlayRecording,
)
from sic_framework.devices.common_naoqi.naoqi_stiffness import Stiffness
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest

import random

class NaoGeminiConversation(SICApplication):
    def __init__(self):
        super(NaoGeminiConversation, self).__init__()

        self.nao_ip = "10.0.0.243"

        self.negative_reactions = ["negative_reactions/motion_stupidx3", "negative_reactions/motion_no_no_no", "negative_reactions/motion_oh_man", "negative_reactions/motion_desperation_and_disappointment"]
        self.positive_reactions = ["positive_reactions/motion_clapping", "positive_reactions/motion_mwak", "positive_reactions/motion_yay", "positive_reactions/motion_yay"]

        self.motion_name = None
        self.chain = None

        self.nao = None
        self.gemini_session = None
        self.loop = None

        self.is_nao_speaking = False  
        self.model_is_speaking = False 
        self.buffered_text = [] 

        self.set_log_level(sic_logging.INFO)
        self.setup()

    def setup(self):
        self.logger.info("Initializing NAO...")
        
        conf = NaoqiMotionRecorderConf(use_sensors=True)
        self.nao = Nao(ip=self.nao_ip, motion_record_conf=conf)

    async def perform_nao_dance(self, style: str | None = None):
        self.logger.info(f"NAO dance requested with style: {style}")

        # {motion:chain} dictionary for different reactions
        motion_chains = {"negative_reactions/motion_stupidx3":['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'], 
                         "negative_reactions/motion_no_no_no":['HeadYaw', 'HeadPitch'],
                         "negative_reactions/motion_oh_man":['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'],
                         "negative_reactions/motion_desperation_and_disappointment":['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch', 'RKneePitch', 'LKneePitch', 'RAnklePitch', 'LAnklePitch', 'RAnkleRoll', 'LAnkleRoll', 'LAnkleRoll', 'RHipYawPitch', 'LHipYawPitch', 'RHipRoll', 'LHipRoll', 'RHipPitch', 'LHipPitch'],
                         "positive_reactions/motion_clapping":['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'],
                         "positive_reactions/motion_mwak":['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'],
                         "positive_reactions/motion_yay":['RKneePitch', 'LKneePitch', 'RAnklePitch', 'LAnklePitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowRoll', 'RElbowYaw', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'],
                         "positive_reactions/motion_yeah":['HeadYaw', 'HeadPitch']}

        if style == "coach": # TODO: change to negative word to trigger negative reactions
            self.logger.info(f"Trigger {style} detected! Replaying specific recording...")
            # Randomly select a negative reaction motion 
            self.motion_name = random.choice(self.negative_reactions)
            self.chain = motion_chains[self.motion_name]
            await asyncio.to_thread(self._execute_replay_logic)

        elif style == "good": 
            self.logger.info(f"Trigger {style} detected! Replaying specific recording...")
            # Randomly select a positive reaction motion 
            self.motion_name = random.choice(self.positive_reactions)
            self.chain = motion_chains[self.motion_name]
            await asyncio.to_thread(self._execute_replay_logic)
        
        else:
            self.logger.info("Standard dance requested.")
            await asyncio.sleep(2)

        self.logger.info("NAO action routine completed.")

    def _execute_replay_logic(self):
        try:
            self.logger.info("Replaying action (Stiffness -> Load -> Play)")
            
            self.nao.stiffness.request(
                Stiffness(stiffness=0.7, joints=self.chain)
            )  
            
            recording = NaoqiMotionRecording.load(self.motion_name)
            self.nao.motion_record.request(PlayRecording(recording))
            
        except Exception as e:
            self.logger.error(f"Error replaying motion: {e}")

    def on_nao_audio(self, message: AudioMessage):
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

    async def run_gemini(self):
        client = genai.Client(api_key=)
        model = "gemini-live-2.5-flash-preview"

        start_dance_tool = {
            "name": "start_dance",
            "description": "Make the NAO social robot perform a motion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "style": {
                        "type": "string",
                        "description": "The style of motion. Use 'coach' if the user says that word.",
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
                "If the user says the word 'Coach' or 'Coach me', you MUST call the "
                "'start_dance' tool with the style set to 'coach'. "
                "For other requests, use 'happy' or 'random'."
            ),
            "tools": [
                {"function_declarations": [start_dance_tool]},
            ],
        }

        async with client.aio.live.connect(model=model, config=config) as session:
            self.gemini_session = session
            self.loop = asyncio.get_running_loop()

            self.nao.mic.register_callback(self.on_nao_audio)
            self.logger.info("Microphone callback registered. Say 'Coach' to test!")

            while not self.shutdown_event.is_set():
                async for response in session.receive():
                    sc = response.server_content

                    if response.tool_call:
                        await self.handle_tool_calls(response)

                    if response.text is not None:
                        if not self.model_is_speaking:
                            self.model_is_speaking = True
                            self.is_nao_speaking = True
                            self.logger.info("Model started responding; mic muted.")

                        self.buffered_text.append(response.text)

                    if sc and sc.generation_complete:
                        full_text = "".join(self.buffered_text).strip()
                        self.buffered_text = []

                        if full_text:
                            self.logger.info(f"Full model response: {full_text}")
                            self.nao.tts.request(
                                NaoqiTextToSpeechRequest(full_text),
                                block=True,
                            )

                        self.logger.info("NAO finished speaking; mic unmuted.")
                        self.model_is_speaking = False
                        self.is_nao_speaking = False

                await asyncio.sleep(0.05)

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