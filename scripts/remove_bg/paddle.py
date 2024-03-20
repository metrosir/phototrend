from utils.datadir import project_dir
from Matting.tools.bg_replace import main


class PaddleSeg:
    def __init__(self, model_path, cfg, image_path, background, save_dir, fg_estimate):
        self.model_path = model_path
        self.cfg = cfg
        self.image_path = image_path
        self.background = background
        self.save_dir = save_dir
        self.fg_estimate = fg_estimate
        self.device = 'gpu'
        self.trimap_path = None

    def __call__(self):
        return main(self)


def remove(image_path):
    import uuid
    import datetime
    import pytz
    import os
    import pathlib

    utc_plus_8 = pytz.timezone('Asia/Shanghai')
    date_dir = datetime.datetime.now(utc_plus_8).strftime('%Y%m%d')
    dir = f'{project_dir}/worker_data/history/paddle_rembg/{date_dir}/{uuid.uuid4().hex[:8]}/'
    input_dir = dir+'input'
    pathlib.Path(input_dir).mkdir(parents=True, exist_ok=True)
    output_dir = dir+'output'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)


    seg = PaddleSeg(f"{project_dir}/repositories/PaddleSeg/Matting/pretrained_models/ppmatting-hrnet_w18-human_1024.pdparams",
                f"{project_dir}/repositories/PaddleSeg/Matting/configs/ppmatting/ppmatting-hrnet_w18-human_1024.yml",
                # "/data1/aigc/phototrend/test/11306_00.jpg",
                image_path,
                'g',
                f"{project_dir}/worker_data/history/paddle_rembg/",
                True)
    ret = seg()
    print("ret:", ret)