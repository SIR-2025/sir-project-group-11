from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging, utils_cv2
from sic_framework.devices.desktop import Desktop
from sic_framework.devices.common_desktop.desktop_camera import DesktopCameraConf
from sic_framework.services.face_recognition_dnn.face_recognition import (
    DNNFaceRecognition,
    DNNFaceRecognitionComponent,
)
from sic_framework.core.message_python2 import CompressedImageMessage, BoundingBoxesMessage
import cv2, queue, numpy as np, time

class DesktopFaceDemo(SICApplication):
    def __init__(self):
        super().__init__()
        self.set_log_level(sic_logging.INFO)
        self.desktop = None
        self.cam = None
        self.fr = DNNFaceRecognition(input_source=self.cam)
        self.imgs = queue.Queue(maxsize=1)
        self.latest_faces = []
        self.last_frame_ts = 0
        self.last_face_ts = 0
        self.setup()

    def on_image(self, msg: CompressedImageMessage):
        buf = np.frombuffer(msg.image, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None: return
        try: self.imgs.get_nowait()
        except queue.Empty: pass
        self.imgs.put(img)
        self.last_frame_ts = time.time()

    def on_faces(self, msg: BoundingBoxesMessage):
        self.latest_faces = msg.bboxes or []
        self.last_face_ts = time.time()

    def setup(self):
        cam_conf = DesktopCameraConf(fx=1.0, fy=1.0, flip=1)
        self.desktop = Desktop(camera_conf=cam_conf)
        self.cam = self.desktop.camera
        self.fr = DNNFaceRecognition(input_source=self.cam)
        self.cam.register_callback(self.on_image)
        self.fr.register_callback(self.on_faces)
        self.desktop.start()
        self.cam.start()
        self.fr.start()

    def run(self):
        cv2.namedWindow("Desktop Face Recognition", cv2.WINDOW_NORMAL)
        try:
            while not self.shutdown_event.is_set():
                try:
                    frame = self.imgs.get(timeout=0.2)
                except queue.Empty:
                    continue
                for bb in self.latest_faces:
                    utils_cv2.draw_bbox_on_image(bb, frame)
                    if getattr(bb, "meta", None):
                        label = bb.meta.get("name") or bb.meta.get("id")
                        if label:
                            utils_cv2.draw_label_on_image(label, bb, frame)
                cv2.imshow("Desktop Face Recognition", frame)
                cv2.waitKey(1)
        finally:
            cv2.destroyAllWindows()
            for comp in (self.fr, self.cam, self.desktop):
                try: comp.stop()
                except: pass
            self.shutdown()

if __name__ == "__main__":
    DesktopFaceDemo().run()
