import gc
import math
import os
import platform
import PIL.Image
import PIL.ImageOps
import requests
from typing import Union
from utils.cmd_args import opts as shared
from utils.image import auto_resize_to_pil, read_image_to_np
from utils.datadir import generate_inpaint_image_dir
import scripts.devices as devices

from PIL import Image, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo

if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import re
import traceback

import scripts.version as ia_check_versions

import cv2
import gradio as gr
import numpy as np
import torch
from utils.pt_logging import ia_logging
from diffusers import (DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler,
                       StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel)




# def load_im
def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            print(image)
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def run_inpaint(input_image,mask_image, prompt, n_prompt, ddim_steps, cfg_scale, seed, inp_model_id, composite_chk,
                controlnets = [], sampler_name="DDIM", iteration_count=1):

    if platform.system() == "Darwin":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16
    input_image = read_image_to_np(input_image)
    mask_image = read_image_to_np(mask_image)

    controlnet_images = []
    controlnet_scale = []
    controlnet = []
    if len(controlnets) > 0:
        for contr_info in controlnets:
            # contr_tmp = contr_info.copy()
            controlnet_images.append(load_image(contr_info['image']))
            controlnet_scale.append(contr_info['scale'])
            path = contr_info['model_path']
            contr_info.pop('image')
            contr_info.pop('scale')
            contr_info.pop('model_path')

            controlnet.append(
                ControlNetModel.from_pretrained(path, torch_dtype=torch_dtype, **contr_info)
            )

    local_files_only = True

    cannay_model_path = '/data/aigc/stable-diffusion-webui/extensions/sd-webui-controlnet/models/control_v11p_sd15_canny.pth'
    cannay_model_path = 'lllyasviel/sd-controlnet-canny'
    # cannay_image = '/data/aigc/stable-diffusion-webui/extensions/sd-webui-inpaint-anything/images/canny_1.png'
    # cannay_image = load_image(cannay_image)

    try:
        # controlnet=[
        #     ControlNetModel.from_pretrained(cannay_model_path, torch_dtype=torch_dtype)
        # ]
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(inp_model_id, torch_dtype=torch_dtype, local_files_only=True, controlnet=controlnet)
    except Exception as e:
        ia_logging.error(str(e))
        return
    pipe.safety_checker = None

    ia_logging.info(f"Using sampler {sampler_name}")
    if sampler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "DPM2 Karras":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "DPM2 a Karras":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        ia_logging.info("Sampler fallback to DDIM")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if platform.system() == "Darwin":
        pipe = pipe.to("mps" if ia_check_versions.torch_mps_is_available else "cpu")
        pipe.enable_attention_slicing()
        torch_generator = torch.Generator(devices.cpu)
    else:
        if ia_check_versions.diffusers_enable_cpu_offload() and devices.device != devices.cpu:
            ia_logging.info("Enable model cpu offload")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(devices.device)
        if shared.xformers:
            ia_logging.info("Enable xformers memory efficient attention")
            pipe.enable_xformers_memory_efficient_attention()
        else:
            ia_logging.info("Enable attention slicing")
            pipe.enable_attention_slicing()
        if "privateuseone" in str(getattr(devices.device, "type", "")):
            torch_generator = torch.Generator(devices.cpu)
        else:
            torch_generator = torch.Generator(devices.device)

    print(input_image, mask_image)
    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    output_list = []
    iteration_count = iteration_count if iteration_count is not None else 1
    for count in range(int(iteration_count)):
        gc.collect()
        if seed < 0 or count > 0:
            seed = random.randint(0, 2147483647)

        generator = torch_generator.manual_seed(seed)

        pipe_args_dict = {
            "prompt": prompt,
            "image": init_image,
            "control_image": controlnet_images,
            "controlnet_conditioning_scale": controlnet_scale,
            "width": width,
            "height": height,
            "mask_image": mask_image,
            "num_inference_steps": ddim_steps,
            "guidance_scale": cfg_scale,
            "negative_prompt": n_prompt,
            "generator": generator,
        }

        print(f"pipe_args_dict:{pipe_args_dict}")
        output_image = pipe(**pipe_args_dict).images[0]

        if composite_chk:
            dilate_mask_image = Image.fromarray(cv2.dilate(np.array(mask_image), np.ones((3, 3), dtype=np.uint8), iterations=4))
            output_image = Image.composite(output_image, init_image, dilate_mask_image.convert("L").filter(ImageFilter.GaussianBlur(3)))

        generation_params = {
            "Steps": ddim_steps,
            "Sampler": sampler_name,
            "CFG scale": cfg_scale,
            "Seed": seed,
            "Size": f"{width}x{height}",
            "Model": inp_model_id,
        }

        generation_params_text = ", ".join([k if k == v else f"{k}: {v}" for k, v in generation_params.items() if v is not None])
        prompt_text = prompt if prompt else ""
        negative_prompt_text = "\nNegative prompt: " + n_prompt if n_prompt else ""
        infotext = f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()

        metadata = PngInfo()
        metadata.add_text("parameters", infotext)

        # save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(inp_model_id), str(seed)]) + ".png"
        img_idx = len(os.listdir(generate_inpaint_image_dir))

        # save_name = generate_inpaint_image_dir + f"{img_idx}.png"
        save_name=os.path.join(generate_inpaint_image_dir, f"{img_idx}.png")
        # save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
        output_image.save(save_name, pnginfo=metadata)

        output_list.append(output_image)

        return output_list, max([1, iteration_count - (count + 1)])

