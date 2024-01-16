
import wget
import os
from utils.datadir import project_dir

modes = {
    'https://huggingface.co/metrosir/DWPose/resolve/main/dw-ll_ucoco_384.onnx': f"{project_dir}/repositories/DWPose/ControlNet-v1-1-nightly/annotator/ckpts/dw-ll_ucoco_384.onnx",
    'https://huggingface.co/metrosir/DWPose/resolve/main/yolox_l.onnx': f"{project_dir}/repositories/DWPose/ControlNet-v1-1-nightly/annotator/ckpts/yolox_l.onnx",
}


def download_model():
    for url, path in modes.items():
        if not os.path.exists(path):
            wget.download(url, path)
