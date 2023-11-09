from .utils import project_dir

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
        'prompt': '\n (high_contrast), RAW photo,realistic,dramatic lighting,ultra high res,best quality,high quality,<lora:add_detail:1>',
        'negative_prompt': '(human:1.2),realisticvision-negative-embedding',
    },
    self_innovate_mode: {
        'inpaint_weight': 0.75,
        'ip-adapter_weight': 0.75,
        'lineart_weight': 0.7,
        'prompt': '\n (high_contrast), RAW photo,realistic,dramatic lighting,ultra high res,best quality,high quality',
        'negative_prompt': '(human:1.2),realisticvision-negative-embedding',
    }
}

init_model = {
    'base_mode':'metrosir/realistic',
    'controlnets': [
        {
            'low_cpu_mem_usage': False,
            'device_map': None,
            'model_path': 'metrosir/phototrend',
            "subfolder": 'controlnets/ip-adapter-plus',
            'scale': [],
            'image': None,
            'local_files_only': False
        },
        {
            'low_cpu_mem_usage': False,
            'model_path': 'metrosir/phototrend',
            'subfolder': 'controlnets/lineart-fp16',
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