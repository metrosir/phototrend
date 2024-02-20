# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .wholebody import Wholebody

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self, onnx_det_model_path=None, onnx_pose_model_path=None):

        self.pose_estimation = Wholebody(onnx_det_model_path, onnx_pose_model_path)

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            return draw_pose(pose, H, W)

    def is_person_by_score(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]

            f_cnt = 0
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.45:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1
                        f_cnt += 1

            # 0-近、1-中、2-远
            distance_enum = None
            if f_cnt >= 9:
                return False, distance_enum

            distance_tags = {
                "near": [{8, 11, 16, 17}, {8, 11, 16}, {8, 11, 17}],
                "mid": [{9, 12, 16, 17}, {9, 12, 16}, {9, 12, 17}],
                "far": [{10, 13, 16, 17}, {10, 13, 16}, {10, 13, 17}],
            }
            # distance_near = {8, 11, 16, 17}

            pose_scores = []

            for i in range(len(score)):
                for j in range(len(score[i])):
                    pose_scores.append(score[i][j])

            if distance_tags['far'][0].issubset(pose_scores) or distance_tags['far'][1].issubset(pose_scores) or distance_tags['far'][2].issubset(pose_scores):
                distance_enum = 2
            elif distance_tags['mid'][0].issubset(pose_scores) or distance_tags['mid'][1].issubset(pose_scores) or distance_tags['mid'][2].issubset(pose_scores):
                distance_enum = 1
            elif distance_tags['near'][0].issubset(pose_scores) or distance_tags['near'][1].issubset(pose_scores) or distance_tags['near'][2].issubset(pose_scores):
                distance_enum = 0
            # distance_tags['mid'].issubset(pose_scores)
            # distance_tags['near'].issubset(pose_scores)

            # for arr in distance_tags['far']:
            #     if arr in pose_scores:
            #         distance_enum = 2
            #         break
            # if distance_tags['far'] in pose_scores:
            #     distance_enum = 2
            # elif distance_tags['mid'] in pose_scores:
            #     distance_enum = 1
            # elif distance_tags['near'] in pose_scores:
            #     distance_enum = 0

            if distance_enum is not None:
                return True, distance_enum
            return False, distance_enum

            un_visible = subset < 0.3
            candidate[un_visible] = -1
            return not np.all(subset < 0.3)