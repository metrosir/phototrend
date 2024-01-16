import utils.utils as utils
import cv2
import numpy as np

from PIL import Image
from utils.datadir import project_dir
from annotator.dwpose import DWposeDetector

class Pose:
    def __init__(self, img):
        self.pose_obj = DWposeDetector(f"{project_dir}/repositories/DWPose/ControlNet-v1-1-nightly/annotator/ckpts/yolox_l.onnx",
                              f"{project_dir}/repositories/DWPose/ControlNet-v1-1-nightly/annotator/ckpts/dw-ll_ucoco_384.onnx")
        self.img = img
        if not isinstance(self.img, np.ndarray):
            self.img = cv2.imread(self.img)

    def isNotPerson(self):
        return self.pose_obj.is_person_by_score(self.img)

    def getPose(self):
        pose_img = self.pose_obj(self.img)
        pose_img = Image.fromarray(pose_img)
        return pose_img


