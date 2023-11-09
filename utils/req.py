import requests
import json
import pathlib
import base64
import sys
import os
import copy
import utils.datadir as datadir

from utils.image import image_to_base64
from utils.cmd_args import opts

# 'http://10.61.158.18:7860'
sd_host = opts.sd_host
api_txt2img = f'{sd_host}/sdapi/v1/txt2img'
api_interrogate = f'{sd_host}/sdapi/v1/interrogate'
api_sd_models = f'{sd_host}/sdapi/v1/sd-models'
api_sd_vae = f'{sd_host}/sdapi/v1/sd-vae'


def requestsd(url, data, headers=None, method='post'):
    if headers is None:
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

    try:
        if method == 'post':
            response = requests.post(url, headers=headers, data=json.dumps(data))
        else:
            response = requests.get(url, headers=headers, data=json.dumps(data))
    except Exception as e:
        print("requests error:", str(e))
        raise Exception(f"requests error, api:{url}")
    response_data = response.json()
    if "error" not in response_data:
        return response_data
    else:
        return False


def interrogate(image_path):
    data={
        'image': image_to_base64(image_path),
        'model': 'clip'
    }
    response_data = requestsd(api_interrogate, data=data)
    if "caption" in response_data:
        return response_data['caption']
    return None

def sd_models():
    try:
        response_data = requestsd(api_sd_models, data={}, method='get')
    except Exception as e:
        print("requests error:", str(e))
        return None

    if response_data is not False:
        return response_data
    return None


def sd_vae():
    try:
        response_data = requestsd(api_sd_vae, data={}, method='get')
    except Exception as e:
        print("requests error:", str(e))
        return None
    if response_data is not False:
        response_data.insert(0, {'model_name': 'Automatic', 'filename': ''})
        response_data.insert(1, {'model_name': 'None', 'filename': ''})
        return response_data
    return None


prompt = "\n RAW photo,photorealistic,(photorealistic:1.4),ultra high res,best quality,high quality,film grain,Fujifilm XT3,(8k),<lora:add_detail:1>"
prompt = "\n RAW photo,photorealistic,(dramatic lighting:1.4),ultra high res,best quality,high quality,film grain,Fujifilm XT3,(8k),<lora:add_detail:1>"
# high_contrast:1.2 高对比度「」
prompt = "\n (high_contrast:1.2), vivid photo effect, RAW photo,realistic,(dramatic lighting:1.2),ultra high res,best quality,high quality,<lora:add_detail:1>"
prompt = "\n (high_contrast), RAW photo,realistic,dramatic lighting,ultra high res,best quality,high quality,<lora:add_detail:1>"

negative_prompt = '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime),text,cropped,out of frame,worst quality,low quality,jpeg artifacts,ugly,duplicate,morbid,mutilated,extra fingers,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,blurry,dehydrated,bad anatomy,bad proportions,extra limbs,cloned face,disfigured,gross proportions,malformed limbs,missing arms,missing legs,extra arms,extra legs,fused fingers,too many fingers,long neck,UnrealisticDream,'
negative_prompt = '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4),(deformed, distorted, disfigured:1.3),poorly drawn,bad anatomy,wrong anatomy,extra limb,missing limb,floating limbs,disconnected limbs,mutation,mutated,ugly,disgusting,amputation,badhandv4,'
negative_prompt = '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation, UnrealisticDream'

negative_prompt = '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream'
negative_prompt = 'BadDream, (UnrealisticDream:1.2)'
negative_prompt = 'nsfw,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4),(deformed, distorted, disfigured:1.3),poorly drawn,bad anatomy,wrong anatomy,extra limb,missing limb,floating limbs,disconnected limbs,mutation,mutated,ugly,disgusting,amputation,badhandv4'
negative_prompt = '(human:1.2),realisticvision-negative-embedding'
negative_prompt_clothes = 'realisticvision-negative-embedding'


def generate_image(select_model,select_vae, prompt, negative_prompt, batch_count, contr_inp_weight=0.5, contr_ipa_weight=0.55, contr_lin_weight=0.7, g_type=1, width=768, height=1024):
    print(width, height, prompt)
    if int(g_type) == 1:
        comm_merge_scene_im = f'{datadir.commodity_merge_scene_image_dir}/{datadir.get_file_idx()}.png'
        comm_merge_scene_mask_im = f'{datadir.mask_image_dir}/{datadir.get_file_idx()}.png'
    else:
        comm_merge_scene_im = f'{datadir.clothes_merge_scene_dir}/{datadir.get_file_idx(check_dir=datadir.clothes_dir)}.png'
        comm_merge_scene_mask_im = f'{datadir.clothes_mask_dir}/{datadir.get_file_idx(check_dir=datadir.clothes_dir)}.png'

    def mask_invert(mask_img, mask_img_invert):
        # pip install Pillow
        from PIL import Image
        # pip install numpy
        import numpy as np

        # 打开图片
        img = Image.open(mask_img).convert('L')
        # 将图片转换为numpy数组
        img_np = np.array(img)
        # 对图片进行反转
        img_np = 255 - img_np
        # 将反转后的numpy数组转回图片
        img_inverted = Image.fromarray(img_np)
        # 保存反转后的图片
        img_inverted.save(mask_img_invert)

    data = {
        "outpath_samples": "./output",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "sampler_index": "DPM++ SDE Karras",
        "steps": 25,
        "width": width,
        "height": height,
        "seed": -1,
        "cfg_scale": 7,
        "n_iter": batch_count,
        # "restore_faces": True,
        "alwayson_scripts": {},
        "override_settings": {
            # "sd_model_checkpoint": "civitai/SG_161222/Realistic_Vision_V5.1.safetensors [00445494c8]"
            # "sd_model_checkpoint": "civitai/Lykon/absoluterealityv1.K9Gm.safetensors"
            "sd_model_checkpoint": select_model,
            "sd_vae": select_vae
        },
        # "enable_hr": True,
        # ---
        # "hr_scale": 1.2,
        # # 仅用于latent upscalers,该参数与image-to-image含义相同，它控制在之星hires采样步骤之前添加到潜空间中的噪声，必须大于0.5，否则会尝产生模糊的图像，使用latent的好处是没有像esrgan这种一样可能引入放大伪像，sd的解码器生成图像，确保风格一致，缺点是在一定程度改变图像，这取决于去噪强度的值
        # ---
        # "denoising_strength": 0.1,
        "hr_upscaler": "R-ESRGAN 4x+",
        # #采样步数
        # "hr_second_pass_steps":0,
        # # 制定尺寸
        # "hr_resize_x":512,
        # "hr_resize_y":512,
        # "denoising_strength": 0,
        "firstphase_width": 0,
        "firstphase_height": 0,
        # 只返回生成的最终图片(send_images,save_images)?
        "send_images": True,
        "save_images": False,
    }

    controlNetArgsTemp = {
        "enabled": True,
        "module": "",
        "model": "",
        "weight": 1,
        "image": "",
        "mask": "",
        # 0:Balanced,1:My prompt is more important,2:ControlNet is more important
        "control_mode": 0,
        "guidance_start": 0,
        "guidance_end": 0.8,
        # 反转图片
        "invert_image": False,
        # #["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]
        "resize_mode": 1,
        "rgbbgr_mode": False,
        "lowvram": False,
        # 完美匹配想输
        "pixel_perfect": True,
        # "processor_res": 0,
        # "threshold_a": 64,
        # "threshold_b": 64,
        # "guessmode": False
    }

    controlNetArgs = {
        "args": []
    }

    gener_images_path = './output'
    # input_image = f'./input_image/shangping/{shangping}'

    if int(g_type) == 1:
        if not pathlib.Path(datadir.merge_after_mask_cut_image_dir).exists():
            pathlib.Path(datadir.merge_after_mask_cut_image_dir).mkdir(parents=True, exist_ok=True)
        comm_merge_scene_mask_cut_im = f'{datadir.merge_after_mask_cut_image_dir}/{datadir.get_file_idx()}.png'
        mask_invert(comm_merge_scene_mask_im, comm_merge_scene_mask_cut_im)
    else:
        if not pathlib.Path(datadir.clothes_mask_cut_dir).exists():
            pathlib.Path(datadir.clothes_mask_cut_dir).mkdir(parents=True, exist_ok=True)
        comm_merge_scene_mask_cut_im = f'{datadir.clothes_mask_cut_dir}/{datadir.get_file_idx(check_dir=datadir.clothes_dir)}.png'
        mask_invert(comm_merge_scene_mask_im, comm_merge_scene_mask_cut_im)

    # if not pathlib.Path(comm_merge_scene_mask_cut_im).exists():

    pathlib.Path(gener_images_path).mkdir(parents=True, exist_ok=True)
    controlnet_openpose_args = dict(controlNetArgsTemp)
    controlnet_lineart_args = dict(controlNetArgsTemp)
    controlnet_scribble_args = dict(controlNetArgsTemp)

    controlnet_inpaint_args = dict(controlNetArgsTemp)
    controlnet_depth_args = dict(controlNetArgsTemp)
    controlnet_softEdge_args = dict(controlNetArgsTemp)
    controlnet_normal_args = dict(controlNetArgsTemp)
    controlent_ipAdapter_args = dict(controlNetArgsTemp)

    ### inpaint
    controlnet_inpaint_args['model'] = 'control_v11p_sd15_inpaint [ebff9138]'
    controlnet_inpaint_args['module'] = 'inpaint_only+lama'
    controlnet_inpaint_args['guidance_end'] = 1
    # 不带人
    controlnet_inpaint_args['weight'] = contr_inp_weight
    # 带人
    # controlnet_inpaint_args['weight'] = 0.9
    controlnet_inpaint_args['control_mode'] = 0
    controlnet_inpaint_args['image'] = image_to_base64(comm_merge_scene_im)
    controlnet_inpaint_args['mask'] = image_to_base64(comm_merge_scene_mask_cut_im)
    # controlnet_inpaint_args['preprocessor_preview'] = False

    controlent_ipAdapter_args['model'] = 'ip-adapter_sd15_plus [32cd8f7f]'
    controlent_ipAdapter_args['module'] = 'ip-adapter_clip_sd15'
    controlent_ipAdapter_args['guidance_end'] = 1
    controlent_ipAdapter_args['weight'] = contr_ipa_weight
    controlent_ipAdapter_args['control_mode'] = 2
    controlent_ipAdapter_args['image'] = image_to_base64(comm_merge_scene_im)

    controlnet_lineart_args['model'] = 'control_v11p_sd15_lineart [43d4be0d]'
    controlnet_lineart_args['module'] = 'lineart_realistic'
    controlnet_lineart_args['guidance_end'] = 1
    controlnet_lineart_args['weight'] = contr_lin_weight
    controlnet_lineart_args['control_mode'] = 2
    controlnet_lineart_args['image'] = image_to_base64(comm_merge_scene_im)

    controlnet_scribble_args = copy.deepcopy(controlNetArgsTemp)
    controlnet_scribble_args['model'] = 'control_v11p_sd15_scribble [d4ba51ff]'
    controlnet_scribble_args['module'] = 'scribble_xdog'
    controlnet_scribble_args['guidance_end'] = 1
    controlnet_scribble_args['weight'] = 0.7
    controlnet_scribble_args['control_mode'] = 2
    controlnet_scribble_args['image'] = image_to_base64(comm_merge_scene_im)

    if controlnet_inpaint_args['image'] is False or controlnet_inpaint_args['mask'] is False:
        print("error: input image data error")
        sys.exit(1)

    ##模式
    controlNetArgs['args'].append(controlnet_inpaint_args)
    controlNetArgs['args'].append(controlent_ipAdapter_args)
    controlNetArgs['args'].append(controlnet_lineart_args)
    # controlNetArgs['args'].append(controlnet_scribble_args)

    data['alwayson_scripts'] = {
        'ControlNet': controlNetArgs
    }

    response_data = requestsd(api_txt2img, data=data)
    generate_imgs = []

    if int(g_type) == 1:
        idx = datadir.get_file_idx(check_dir=datadir.commodity_merge_scene_image_dir)
        generate_image_sub_dir = datadir.generate_image_dir.format(uuid=datadir.uuid, idx=idx)
    else:
        idx = datadir.get_file_idx(check_dir=datadir.clothes_dir)
        generate_image_sub_dir = datadir.clothes_generate_image_dir.format(uuid=datadir.uuid, idx=idx)

    if not pathlib.Path(generate_image_sub_dir).exists():
        pathlib.Path(generate_image_sub_dir).mkdir(parents=True, exist_ok=True)

    im_name_idx = len(os.listdir(generate_image_sub_dir))
    if "error" not in response_data and "images" in response_data:
        for key, value in response_data.items():
            if key == "images" and isinstance(value, list):
                for idx, v in enumerate(value):
                    if idx <= batch_count - 1:
                        img_data = base64.b64decode(v)
                        name = f'{generate_image_sub_dir}/{im_name_idx + idx}.png'
                        generate_imgs.append(name)
                        with open(name, 'wb') as f:
                            f.write(img_data)
        return generate_imgs
    else:
        return None