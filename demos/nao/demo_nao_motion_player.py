# Import basic preliminaries
from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging

# Import the device(s) we will be using
from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest, NaoWakeUpRequest

# Import message types and requests
from sic_framework.devices.common_naoqi.naoqi_motion_recorder import (
    NaoqiMotionRecorderConf,
    NaoqiMotionRecording,
    PlayRecording,
    StartRecording,
    StopRecording,
)
from sic_framework.devices.common_naoqi.naoqi_stiffness import Stiffness

# Import libraries necessary for the demo
import time
import random

class NaoMotionRecorderDemo(SICApplication):
    """
    NAO motion recorder demo application.
    Demonstrates how to record and replay a motion on a NAO robot.
    """
    
    def __init__(self):
        # Call parent constructor (handles singleton initialization)
        super(NaoMotionRecorderDemo, self).__init__()
        
        # Demo-specific initialization
        self.nao_ip = "10.0.0.241"

        # self.negative_reactions = ["negative_reactions/motion_stupidx3", "negative_reactions/motion_no_no_no", "negative_reactions/motion_oh_man", "negative_reactions/motion_desperation_and_disappointment"]
        # self.positive_reactions = ["positive_reactions/motion_clapping", "positive_reactions/motion_mwak", "positive_reactions/motion_yay", "positive_reactions/motion_yay"]
        # self.all_reactions = self.negative_reactions + self.positive_reactions
        
        self.motion_name = "yay_alt2"

        self.chain = None        
        self.nao = None

        self.set_log_level(sic_logging.INFO)
        
        # Log files will only be written if set_log_file is called. Must be a valid full path to a directory.
        # self.set_log_file("/Users/apple/Desktop/SAIL/SIC_Development/sic_applications/demos/nao/logs")
        self.set_log_file("logs")
        self.setup()
    
    def setup(self):
        """Initialize and configure the NAO robot."""
        self.logger.info("Starting NAO Motion Player Demo...")
        
        # Initialize NAO with motion recorder configuration
        conf = NaoqiMotionRecorderConf(use_sensors=True)
        self.nao = Nao(self.nao_ip, motion_record_conf=conf)
    
    def run(self):
        """Main application logic."""
        try:

            self.nao.autonomous.request(NaoWakeUpRequest())

            # select a random motion from the list  
            motion_chains = {"negative_reactions/motion_stupidx3":['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'], 
                            "negative_reactions/motion_no_no_no":['HeadYaw', 'HeadPitch'],
                            "negative_reactions/motion_oh_man":['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'],
                            "negative_reactions/motion_desperation_and_disappointment":['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch', 'RKneePitch', 'LKneePitch', 'RAnklePitch', 'LAnklePitch', 'RAnkleRoll', 'LAnkleRoll', 'RHipYawPitch', 'LHipYawPitch', 'RHipRoll', 'LHipRoll', 'RHipPitch', 'LHipPitch'],
                            "positive_reactions/motion_clapping":['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'],
                            "positive_reactions/motion_mwak":['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'],
                            "positive_reactions/motion_yay":['RKneePitch', 'LKneePitch', 'RAnklePitch', 'LAnklePitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowRoll', 'RElbowYaw', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'],
                            "positive_reactions/motion_yeah":['HeadYaw', 'HeadPitch'],
                            "positive_reactions/motion_handsup_excited":['RKneePitch', 'LKneePitch', 'RAnklePitch', 'LAnklePitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'RShoulderPitch', 'RShoulderRoll', 'RElbowRoll', 'RElbowYaw', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'],
                            "neutral_reactions/motion_present_left_team":['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand', 'HeadYaw', 'HeadPitch'],
                            "neutral_reactions/motion_present_right_team":['RShoulderPitch', 'RShoulderRoll', 'RElbowRoll', 'RElbowYaw', 'RWristYaw', 'RHand', 'HeadYaw', 'HeadPitch'],}
            
            # self.motion_name = random.choice(self.all_reactions)
            self.chain = motion_chains[self.motion_name]
            self.logger.info("Replaying action")
            self.nao.stiffness.request(
                Stiffness(stiffness=0.7, joints=self.chain)
            )
            recording = NaoqiMotionRecording.load(self.motion_name)
            self.nao.motion_record.request(PlayRecording(recording))

            # always end with a rest, whenever you reach the end of your code
            self.nao.autonomous.request(NaoRestRequest())
            self.logger.info("Motion player demo completed successfully")
        except Exception as e:
            self.logger.error("Exception: {}".format(e=e))
        finally:
            self.shutdown()


if __name__ == "__main__":
    # Create and run the demo
    demo = NaoMotionRecorderDemo()
    demo.run()