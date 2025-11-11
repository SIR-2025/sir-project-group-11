import time
import json
import numpy as np
from os.path import abspath, join

from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.services.dialogflow_cx.dialogflow_cx import (
    DialogflowCX,
    DialogflowCXConf,
    DetectIntentRequest,
)


class NaoConversationStarterCX(SICApplication):
    """
    Continuous NAO + Dialogflow CX listener.
    Listens for simple football-related intents and responds safely.
    """

    def __init__(self, nao_ip, google_keyfile_path, agent_id, location):
        super(NaoConversationStarterCX, self).__init__()
        self.nao_ip = nao_ip
        self.google_keyfile_path = google_keyfile_path
        self.agent_id = agent_id
        self.location = location

        self.nao = None
        self.dialogflow_cx = None
        self.session_id = np.random.randint(10000)

        self.set_log_level(sic_logging.INFO)
        self.setup()

    def setup(self):
        """Initialize and configure NAO robot and Dialogflow CX."""
        self.logger.info("Initializing NAO robot...")
        self.nao = Nao(ip=self.nao_ip, dev_test=True)
        nao_mic = self.nao.mic

        self.logger.info("Initializing Dialogflow CX...")

        with open(self.google_keyfile_path) as f:
            keyfile_json = json.load(f)

        dialogflow_conf = DialogflowCXConf(
            keyfile_json=keyfile_json,
            agent_id=self.agent_id,
            location=self.location,
            sample_rate_hertz=16000,  # NAO mic sample rate
            language="en",
        )

        self.dialogflow_cx = DialogflowCX(conf=dialogflow_conf, input_source=nao_mic)

    def run(self):
        """Main continuous listening loop."""
        try:
            self.nao.tts.request(
                NaoqiTextToSpeechRequest("Hello! I am ready to play football with you.")
            )
            time.sleep(0.5)
            self.logger.info("Ready to detect intents...")

            while not self.shutdown_event.is_set():
                self.logger.info("Listening for user command...")

                # Add small delay to avoid overlap from previous TTS
                time.sleep(0.3)

                # Request intent detection (single-utterance)
                reply = self.dialogflow_cx.request(DetectIntentRequest(self.session_id))

                if not reply:
                    self.logger.warning("No reply received from Dialogflow.")
                    continue

                # Log transcript and confidence
                transcript = getattr(reply, "transcript", "")
                intent = getattr(reply, "intent", None)
                conf = getattr(reply, "intent_confidence", None)
                self.logger.info(f"User said: {transcript}")
                self.logger.info(f"Detected intent: {intent}, confidence: {conf}")

                # Ignore low-confidence or empty detections
                if not intent or (conf is not None and conf < 0.7):
                    self.logger.info("Low confidence or no intent detected.")
                    continue

                # Safely mute mic during TTS (if supported)
                if hasattr(self.nao.mic, "mute"):
                    self.nao.mic.mute(True)

                # Intent-specific actions
                if intent == "fetch_ball":
                    self.nao.tts.request(
                        NaoqiTextToSpeechRequest("Okay! I will fetch the ball.")
                    )

                elif intent == "pass_ball":
                    self.nao.tts.request(
                        NaoqiTextToSpeechRequest("Here you go, passing the ball.")
                    )

                elif intent == "score_goal":
                    self.nao.tts.request(
                        NaoqiTextToSpeechRequest("Let's see if I can score a goal!")
                    )

                elif intent == "celebrate":
                    self.nao.tts.request(
                        NaoqiTextToSpeechRequest("Yay! I scored a goal!")
                    )

                elif intent == "exit":
                    self.nao.tts.request(NaoqiTextToSpeechRequest("Goodbye!"))
                    break

                else:
                    self.nao.tts.request(
                        NaoqiTextToSpeechRequest("I'm not sure what you mean.")
                    )

                # Wait for NAO to finish speaking, then unmute
                time.sleep(0.8)
                if hasattr(self.nao.mic, "mute"):
                    self.nao.mic.mute(False)

                # Add brief pause before next round
                time.sleep(0.4)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        except Exception as e:
            import traceback

            self.logger.error(f"Exception occurred: {e}")
            traceback.print_exc()
            self.nao.tts.request(
                NaoqiTextToSpeechRequest(
                    "I encountered an error. Please check the logs."
                )
            )
        finally:
            self.logger.info("Shutting down Dialogflow listener.")
            self.shutdown()


if __name__ == "__main__":
    # Update these values for your setup
    NAO_IP = "10.0.0.241"
    AGENT_ID = "84a97385-2b9e-4186-9660-54f57b21e6d8"
    LOCATION = "europe-west4"

    # Path to your Google Cloud key file
    KEYFILE_PATH = abspath(join("conf", "google", "google-key.json"))

    demo = NaoConversationStarterCX(
        nao_ip=NAO_IP,
        google_keyfile_path=KEYFILE_PATH,
        agent_id=AGENT_ID,
        location=LOCATION,
    )
    demo.run()
