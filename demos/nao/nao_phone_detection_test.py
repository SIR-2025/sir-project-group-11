from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging, utils_cv2
from sic_framework.core.message_python2 import BoundingBoxesMessage, CompressedImageMessage
from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_camera import NaoqiCameraConf
from sic_framework.services.object_detection.object_detection import ObjectDetection, ObjectDetectionConf
from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest, NaoLEDRequest

import queue
import cv2
import time


class NaoPhoneDetectionDemo(SICApplication):
    """Detect phones with NAO camera and change LEDs."""

    def __init__(self):
        super().__init__()

        # NAO config
        self.nao_ip = "10.0.0.241"
        self.nao = None

        # Buffers
        self.imgs_buffer = queue.Queue(maxsize=1)
        self.latest_objects = []

        # Detection
        self.object_det = None

        self.phone_detected = False  # track state to avoid repeated LED requests

        self.set_log_level(sic_logging.INFO)
        self.setup()

    def on_image(self, image_message: CompressedImageMessage):
        img = image_message.image
        try:
            self.imgs_buffer.get_nowait()
        except queue.Empty:
            pass
        self.imgs_buffer.put(img)

    def on_objects(self, message: BoundingBoxesMessage):
        self.latest_objects = message.bboxes

        # Check if a certain is detected
        phone_found = any(obj.identifier.lower() in ["cell phone", "phone"] for obj in self.latest_objects)
        if phone_found and not self.phone_detected:
            self.phone_detected = True
            self.logger.info("Phone detected! Turning LEDs green...")
            self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 1, 0, 0))
        elif not phone_found and self.phone_detected:
            self.phone_detected = False
            self.logger.info("Phone not detected. Turning LEDs off...")
            self.nao.leds.request(NaoLEDRequest("FaceLeds", False))

        # # Move toward object if detected
        # if phone_found:
        #     phone_obj = next((o for o in self.latest_objects if o.identifier.lower() in ["cell phone", "phone"]), None)
        #     if phone_obj:
        #         x_center = phone_obj.x + phone_obj.w / 2
        #         img_center = 640 / 2
        #         offset = x_center - img_center

        #         turn_angle = -0.002 * offset
        #         self.nao.motion.turn(turn_angle)

        #         desired_box_height = 300
        #         distance = 0.001 * (desired_box_height - phone_obj.h)
        #         if distance > 0:
        #             self.nao.motion.walk_forward(distance)

    def setup(self):
        self.logger.info("Initializing NAO...")
        conf = NaoqiCameraConf(vflip=1)
        self.nao = Nao(ip=self.nao_ip, top_camera_conf=conf)

        self.logger.info("Setting up object detection...")
        obj_det_conf = ObjectDetectionConf(frequency=15.0)
        self.object_det = ObjectDetection(input_source=self.nao.top_camera, conf=obj_det_conf)

        self.logger.info("Registering callbacks...")
        self.nao.top_camera.register_callback(self.on_image)
        self.object_det.register_callback(self.on_objects)

    def run(self):
        self.logger.info("Starting main loop...")
        try:
            while not self.shutdown_event.is_set():
                try:
                    img = self.imgs_buffer.get(timeout=0.1)
                    for obj in self.latest_objects:
                        utils_cv2.draw_bbox_on_image(obj, img)
                    cv2.imshow("NAO Phone Detection", img[..., ::-1])
                    cv2.waitKey(1)
                except queue.Empty:
                    continue
            cv2.destroyAllWindows()
        finally:
            cv2.destroyAllWindows()
            self.shutdown()


if __name__ == "__main__":
    demo = NaoPhoneDetectionDemo()
    demo.run()
