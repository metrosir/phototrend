import logging.config
import pathlib

from .utils import project_dir
import datetime

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


def truncate_large_fields(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and len(value) > 1000:
                obj[key] = ''
            else:
                truncate_large_fields(value)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            if isinstance(item, str) and len(item) > 1000:
                obj[idx] = ''
            elif isinstance(item, dict) or isinstance(item, list):
                truncate_large_fields(obj)


def log_echo(title: str, msg: dict, exception: Exception = None, is_collect: bool = False, level: str = "error", path: str = 'phototrend'):
    import requests
    import traceback
    import json
    import threading

    # try:
    #     msg = json.dumps(msg)
    # except:
    #     pass

    if exception is None:
        data = {
            "date": datetime.datetime.now(
                tz=datetime.timezone(datetime.timedelta(hours=8))
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "level": level,
            "env": PT_ENV}
    else:
        traceback_str = traceback.format_exc()
        data = {
            "date": datetime.datetime.now(
                tz=datetime.timezone(datetime.timedelta(hours=8))
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "exception": str(exception),
            "level": level,
            "traceback": str(traceback_str),
            "env": PT_ENV}

    if msg is not None:
        truncate_large_fields(msg)
        for k, v in msg.items():
            data[f"__{k}"] = v
    data_str = ''
    try:
        data_str = json.dumps(data)
    except Exception as e:
        ia_logging.error(f"collect_info error: {e}", exc_info=True)

    if level == "info":
        if path is not None:
            write_file(path, data_str, level=level)
        else:
            ia_logging.info(f"{title}: {data}")
    else:
        if path is not None:
            write_file(path, data_str, level=level)
        else:
            ia_logging.error(f"{title}: {data}", exc_info=True)

    if is_collect:
        def send():
            try:
                requests.put(COLLECT_URL, data=data_str, timeout=0.5)
            except Exception as e:
                ia_logging.error(f"collect_info error: {e}", exc_info=True)
                pass
        thread = threading.Thread(target=send)
        thread.start()


def write_file(path: str, content: str, level: str = "error"):
    path = f"/data/log/phototrend/{path}"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    file_name = f"{path}/{level}.log"
    with open(file_name, 'a') as f:
        f.write(content + '\n')
        f.close()

def draw_text_image(
        input_image: Union[np.ndarray, Image.Image],
        draw_text: str,
        ) -> Image.Image:
    input_image = np.array(input_image) if isinstance(input_image, Image.Image) else input_image
    ret_image = Image.fromarray(np.zeros_like(input_image))
    draw_ret_image = ImageDraw.Draw(ret_image)
    draw_ret_image.text((0, 0), draw_text, fill=(224, 224, 224))

    return ret_image
