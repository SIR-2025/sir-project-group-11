# Import basic preliminaries
from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging

# Import the device(s) we will be using
from sic_framework.devices import Nao

# Import message types and requests
from sic_framework.devices.common_naoqi.naoqi_leds import (
    NaoFadeRGBRequest,
    NaoLEDRequest,
)
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import (
    NaoqiTextToSpeechRequest,
)

# Import libraries necessary for the demo
import time


class NaoLEDEmotionsDemo(SICApplication):
    """
    NAO LED Emotions Demo.
    
    Demonstrates expressing emotions through LED lights (eyes and chest) 
    synchronized with TTS. Each emotion is announced and then displayed 
    via color coding.
    """
    
    def __init__(self):
        # Call parent constructor (handles singleton initialization)
        super(NaoLEDEmotionsDemo, self).__init__()
        
        # Demo-specific initialization
        self.nao_ip = "10.0.0.243"  # Replace with your NAO's IP address
        self.nao = None

        self.set_log_level(sic_logging.INFO)
        
        self.setup()
    
    def setup(self):
        """Initialize and configure the NAO robot."""
        self.logger.info("Initializing NAO...")
        
        # Initialize the NAO robot
        self.nao = Nao(ip=self.nao_ip)
        
    def display_emotion(self, emotion_name, r, g, b, speech_text):
        """
        Helper function to announce an emotion and set the LEDs.
        
        Args:
            emotion_name (str): Name of the emotion (for logging).
            r, g, b (float): RGB values between 0.0 and 1.0.
            speech_text (str): What the robot should say.
        """
        if self.shutdown_event.is_set():
            return

        self.logger.info(f"Displaying emotion: {emotion_name}")
        
        # 1. Announce the emotion
        self.nao.tts.request(NaoqiTextToSpeechRequest(speech_text))
        
        # 2. Change LEDs (Fade duration 0.5s)
        # Colors are R, G, B values from 0 to 1
        
        # Eyes (FaceLeds)
        self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", r, g, b, 0.5))
        # Chest Button (ChestLeds)
        self.nao.leds.request(NaoFadeRGBRequest("ChestLeds", r, g, b, 0.5))
        # Feet (FeetLeds) - if supported by the specific model
        self.nao.leds.request(NaoFadeRGBRequest("FeetLeds", r, g, b, 0.5))

        # Hold the emotion for a few seconds to let the user see it
        time.sleep(3.0)

    def run(self):
        """Main application logic."""
        self.logger.info("Starting LED Emotions Sequence...")
        
        try:
            # Ensure LEDs are enabled
            self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
            self.nao.leds.request(NaoLEDRequest("ChestLeds", True))
            self.nao.leds.request(NaoLEDRequest("FeetLeds", True))

            # Define sequence of emotions: (Name, Red, Green, Blue, Speech)
            emotions = [
                ("Happy", 1.0, 0.6, 0.0, "I am feeling happy!"),        # Gold/Orange
                ("Sad", 0.0, 0.0, 1.0, "I am feeling sad."),            # Blue
                ("Angry", 1.0, 0.0, 0.0, "I am feeling angry!"),        # Red
                ("Surprised", 1.0, 1.0, 1.0, "Oh! I am surprised."),    # White
                ("Love", 1.0, 0.0, 0.5, "I feel so much love."),        # Pink
                ("Calm", 0.0, 1.0, 0.0, "I am feeling calm."),          # Green
                ("Thinking", 0.0, 1.0, 1.0, "Hmm, let me think."),      # Cyan
                ("Confused", 0.5, 0.0, 1.0, "I am a bit confused.")     # Purple
            ]

            # Execute the sequence
            for name, r, g, b, text in emotions:
                self.display_emotion(name, r, g, b, text)

            # Sequence finished
            self.logger.info("Emotion sequence completed.")
            
            # Reset to neutral state (White)
            self.nao.tts.request(NaoqiTextToSpeechRequest("Returning to normal state."))
            self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1.0, 1.0, 1.0, 0.5))
            self.nao.leds.request(NaoFadeRGBRequest("ChestLeds", 1.0, 1.0, 1.0, 0.5))
            self.nao.leds.request(NaoFadeRGBRequest("FeetLeds", 1.0, 1.0, 1.0, 0.5))
            
            # Allow time for the fade
            time.sleep(1.0)

        except Exception as e:
            self.logger.error(f"Error in LED Emotions Demo: {e}")
        finally:
            self.shutdown()


if __name__ == "__main__":
    # Create and run the demo
    demo = NaoLEDEmotionsDemo()
    demo.run()