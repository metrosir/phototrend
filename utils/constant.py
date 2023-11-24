import pathlib

from .utils import project_dir
import os

def_model = {
    "commodity": "icbinpRelapseRC.zgS8.safetensors",
    "commodity_hand": "",
}

def_vae = {
    "commodity": "vae-ft-mse-840000-ema-pruned.safetensors",
    "commodity_hand": "",
}

sd_mode='SDWebui'
self_innovate_mode='Self-Innovate'
# 作图模式
generate_mode = {
    sd_mode: sd_mode,
    self_innovate_mode: self_innovate_mode,
}

# 作图模式对应的权重
mode_params = {
    sd_mode: {
        'inpaint_weight': 0.5,
        'ip-adapter_weight': 0.55,
        'lineart_weight': 0.7,
        'scribble_weight': 0.7,
        'sampler_step':30,
        'sampler_name': 'UniPC',
        'prompt': '\n (high_contrast), RAW photo,realistic,dramatic lighting,ultra high res,best quality,high quality,<lora:add_detail:1>',
        'negative_prompt': '(human:1.2),realisticvision-negative-embedding',
    },
    self_innovate_mode: {
        # 'inpaint_weight': 0.95,
        'inpaint_weight': 0.95,
        'ip-adapter_weight': 0.75,
        # 'lineart_weight': 0.7,
        'lineart_weight': 0.2,
        'scribble_weight': 2,
        'sampler_step':10,
        'sampler_name': 'UniPC',
        # 'prompt': '\n (high_contrast), RAW photo,realistic,dramatic lighting,ultra high res,best quality,high quality',
        'prompt': '\n marble table top,(the enhanced ones have higher saturation of colour), RAW photo,realistic,dramatic lighting,ultra high res,best quality,high quality',
        'negative_prompt': '(human:1.2),realisticvision-negative-embedding',
    }
}

init_model = {
    'base_mode':'metrosir/realistic',
    'controlnets': [
        # {
        #     'low_cpu_mem_usage': False,
        #     'device_map': None,
        #     'model_path': 'metrosir/phototrend',
        #     "subfolder": 'controlnets/ip-adapter-plus',
        #     'scale': [],
        #     'image': None,
        #     'local_files_only': False
        # },
        {
            'low_cpu_mem_usage': False,
            'model_path': 'metrosir/phototrend',
            'subfolder': 'controlnets/lineart-fp16',
            'scale': [],
            'device_map': None,
            'image': None,
        },
        # {
        #     'low_cpu_mem_usage': False,
        #     'model_path': 'metrosir/phototrend',
        #     'subfolder': 'controlnets/lineart',
        #     'scale': [],
        #     'device_map': None,
        #     'image': None,
        # },
        {
            'low_cpu_mem_usage': False,
            'model_path': 'lllyasviel/control_v11p_sd15_scribble',
            'subfolder': '',
            'scale': [],
            'device_map': None,
            'image': None,
        }
    ],
    'textual_inversion': {
        'model_id': f'{project_dir}/models/textual_inversion/negative_prompt/realisticvision-negative-embedding.pt',
        'token': 'realisticvision-negative-embedding',
        'weight_name': 'string_to_param',
    }
}

commodity_type = [
    '化妆品',
    '服装',
]

commodity_shape = [
    '横图',
    '竖图',
]

api_queue_dir = f"{project_dir}/worker_data/api_queue"

COLLECT_URL = os.getenv("COLLECT_URL")

# pttest, ptdev, ptprod
PT_ENV = os.getenv("PT_ENV")
PT_PROD = "pt_prod"

is_prod = False
if PT_ENV == PT_PROD:
    is_prod = True

# 1.1.1.1,2.2.2.2
hosts = os.getenv("PT_BACK_HOSTS")