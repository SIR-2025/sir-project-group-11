# Import basic preliminaries
from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging, utils_cv2
from sic_framework.core.message_python2 import BoundingBoxesMessage, CompressedImageMessage

# Import device and camera
from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_camera import NaoqiCameraConf

# Import object detection
from sic_framework.services.object_detection.object_detection import ObjectDetection, ObjectDetectionConf

# Other libraries
import queue
import cv2


class NaoObjectDetectionDemo(SICApplication):
    """
    NAO object detection using NAOqi camera.
    """

    def __init__(self):
        super(NaoObjectDetectionDemo, self).__init__()

        # NAO configuration
        self.nao_ip = "10.0.0.241"  # Replace with your NAO's IP
        self.nao = None

        # Buffers
        self.imgs_buffer = queue.Queue(maxsize=1)
        self.latest_objects = []

        # Object detection
        self.object_det = None

        # Logging
        self.set_log_level(sic_logging.INFO)

        self.setup()

    def on_image(self, image_message: CompressedImageMessage):
        """Callback for incoming NAO camera images."""
        # Convert NAO image to OpenCV format if necessary
        img = image_message.image
        try:
            self.imgs_buffer.get_nowait()
        except queue.Empty:
            pass
        self.imgs_buffer.put(img)

    def on_objects(self, message: BoundingBoxesMessage):
        """Callback for object detection results."""
        self.latest_objects = message.bboxes
        if self.latest_objects:
            print(vars(self.latest_objects[0]))

    def setup(self):
        """Initialize NAO camera and object detection service."""
        self.logger.info("Initializing NAO...")
        conf = NaoqiCameraConf(vflip=1)
        self.nao = Nao(ip=self.nao_ip, top_camera_conf=conf)

        self.logger.info("Setting up object detection service...")
        obj_det_conf = ObjectDetectionConf(frequency=15.0)  # 15 Hz detection
        self.object_det = ObjectDetection(input_source=self.nao.top_camera, conf=obj_det_conf)

        self.logger.info("Registering callbacks...")
        self.nao.top_camera.register_callback(self.on_image)
        self.object_det.register_callback(self.on_objects)

    def run(self):
        """Main loop: display NAO camera feed with bounding boxes."""
        self.logger.info("Starting main loop...")

        try:
            while not self.shutdown_event.is_set():
                try:
                    img = self.imgs_buffer.get(timeout=0.1)

                    # Draw detected bounding boxes
                    for obj in self.latest_objects:
                        utils_cv2.draw_bbox_on_image(obj, img)

                    cv2.imshow("NAO Object Detection", img[..., ::-1])  # NAO is RGB, OpenCV expects BGR
                    cv2.waitKey(1)
                except queue.Empty:
                    continue

            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.error(f"Exception: {e}")
        finally:
            cv2.destroyAllWindows()
            self.shutdown()


if __name__ == "__main__":
    demo = NaoObjectDetectionDemo()
    demo.run()
