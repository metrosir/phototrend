import logging.config
from .utils import project_dir

import warnings
from typing import Union
from utils.constant import COLLECT_URL, PT_ENV

import numpy as np
from PIL import Image, ImageDraw
import asyncio

warnings.filterwarnings(action="ignore", category=FutureWarning, module="transformers")

logging.config.fileConfig(f'{project_dir}/configs/log/logging.conf')

ia_logging = logging.getLogger("PhotoTrend")
ia_logging.setLevel(logging.INFO)
ia_logging.propagate = False

ia_logging_sh = logging.StreamHandler()
ia_logging_sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
ia_logging_sh.setLevel(logging.INFO)
ia_logging.addHandler(ia_logging_sh)


def w_info(title, msg):
    pass

def w_debug(msg: str):
    pass
def w_error(msg: str):
    pass


def log_echo(title: str, msg: object, exception: Exception = None, is_collect: bool = False, level: str = "error"):
    import requests
    import traceback
    import json
    import threading

    try:
        msg = json.dumps(msg)
    except:
        pass

    if exception is None:
        data = {"title": title,
                "msg": msg,
                "level": level,
                "env": PT_ENV}
    else:
        traceback_str = traceback.format_exc()
        data = {"title": title,
                "msg": msg,
                "exception": str(exception),
                "level": level,
                "traceback": traceback_str,
                "env": PT_ENV}
    if level == "info":
        ia_logging.info(f"{title}: {data}")
    else:
        ia_logging.error(f"{title}: {data}", exc_info=True)
    if is_collect:
        def send():
            try:
                requests.put(COLLECT_URL, data=json.dumps(data), timeout=0.5)
            except Exception as e:
                ia_logging.error(f"collect_info error: {e}", exc_info=True)
                pass
        thread = threading.Thread(target=send)
        thread.start()

def draw_text_image(
        input_image: Union[np.ndarray, Image.Image],
        draw_text: str,
        ) -> Image.Image:
    input_image = np.array(input_image) if isinstance(input_image, Image.Image) else input_image
    ret_image = Image.fromarray(np.zeros_like(input_image))
    draw_ret_image = ImageDraw.Draw(ret_image)
    draw_ret_image.text((0, 0), draw_text, fill=(224, 224, 224))

    return ret_image
