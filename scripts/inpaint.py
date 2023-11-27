import gc
import math
import os
import pathlib
import platform
import PIL.Image
import PIL.ImageOps
import requests
from typing import Union
from utils.cmd_args import opts as shared
from utils.image import auto_resize_to_pil, read_image_to_np, encode_to_base64, save_output_image_to_pil
from utils.datadir import generate_inpaint_image_dir
import scripts.devices as devices

from PIL import Image, ImageFilter
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
from diffusers import (DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,UniPCMultistepScheduler,
                       KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler,
                       StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel)
from diffusers.pipelines.controlnet import MultiControlNetModel

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_warning()
import warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at metrosir/phototrend were not used when initializing ControlNetModel: ['ip_adapter', 'image_proj']")
from typing import List



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
    elif sampler_name == "UniPC":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
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
    pipe.text_encoder = None
    # print(input_image, mask_image)
    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    output_list = []
    iteration_count = iteration_count if iteration_count is not None else 1
    for count in range(int(iteration_count)):
        gc.collect()
        if seed < 0 or count > 0:
            seed = random.randint(0, 2147483647)

        generator = torch_generator.manual_seed(seed)

        # generator = torch.Generator(device="cpu").manual_seed(1)
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
            "strength": 0.5,
            "eta": 0.1,
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

gpipe=None


class Inpainting:
    pipe = None

    def __init__(self, base_model, subfolder, controlnet: list, textual_inversion: dict):
        self.base_model = base_model
        self.base_model_sub = subfolder
        self.controlnet_image = []
        self.controlnet_scale = []

        self.controlnets = controlnet
        self.local_files_only = False
        if platform.system() == "Darwin":
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch.float16
        self.pipe = self.setup()
        self.set_textual_inversion(**textual_inversion)

        self.pipe.safety_checker = None
        self.sampler_name = None
        self.ipadpter_model = None
        self.device='cuda'

        self.prompt_embeds=None
        self.negative_prompt_embeds=None

    def setup(self):
        try:
            controlnet = []
            for contr_info in self.controlnets:
                path = contr_info['model_path']
                contr_info.pop('image')
                contr_info.pop('scale')
                contr_info.pop('model_path')
                controlnet.append(
                    ControlNetModel.from_pretrained(path, torch_dtype=self.torch_dtype, safety_checker=None, **contr_info)
                )
            # from diffusers import AutoencoderKL
            # vae_model_path = '/data/aigc/diffusers/scripts/diffusers_model/'
            # vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
            # noise_scheduler = DDIMScheduler(
            #     num_train_timesteps=1000,
            #     beta_start=0.00085,
            #     beta_end=0.012,
            #     beta_schedule="scaled_linear",
            #     clip_sample=False,
            #     set_alpha_to_one=False,
            #     steps_offset=1,
            # )
            self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.base_model,
                # vae=vae,
                torch_dtype=self.torch_dtype,
                local_files_only=self.local_files_only,
                controlnet=controlnet,
                # scheduler=noise_scheduler,
                # subfolder=self.base_model_sub,
            )
        except Exception as e:
            ia_logging.error(str(e))
            raise ValueError(str(e))
        return self.pipe



    def set_textual_inversion(self, model_id, token, weight_name, subfolder=None):
        from .piplines.textual_inversion import load
        load(pipe=self.pipe, model_id=model_id, token=token, weight_name=weight_name, subfolder=subfolder)

    def set_controlnet_input(self, controlnets):
        self.controlnet_image = []
        self.controlnet_scale = []
        if len(controlnets) > 0:
            for contr_info in controlnets:
                self.controlnet_image.append(load_image(contr_info['image']))
                self.controlnet_scale.append(contr_info['scale'])
            return self.pipe
        raise ValueError("Controlnet is empty")

    def set_scheduler(self, sampler_name):
        ia_logging.info(f"Using sampler {self.sampler_name}")
        if sampler_name == "DDIM":
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "Euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "Euler a":
            self.pipe.scheduler = \
                EulerAncestralDiscreteScheduler.from_config(
                    self.pipe.scheduler.config
                )
        elif sampler_name == "DPM2 Karras":
            self.pipe.scheduler = KDPM2DiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "UniPC":
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif sampler_name == "DPM2 a Karras":
            self.pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        else:
            ia_logging.info("Sampler fallback to DDIM")
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        return self.pipe

    def set_ip_adapter(self, image_encoder_path, ip_ckpt, device='cuda', num_tokens=16):
        import ip_adapter.ip_adapter
        self.ipadpter_model = ip_adapter.ip_adapter.IPAdapterPlus(self.pipe, image_encoder_path, ip_ckpt, device, num_tokens=num_tokens)


    def input_ip_adapter_condition(self, pil_image, prompt, negative_prompt,clip_image_embeds=None, num_samples=4, scale=1.0):
        self.ipadpter_model.set_scale(scale)
        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        image_prompt_embeds, uncond_image_prompt_embeds = self.ipadpter_model.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            self.prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            self.negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        # self.pipe = ip_model.pipe

    def run_inpaint(self,
                    input_image,mask_image,
                    prompt, n_prompt,
                    ddim_steps, cfg_scale, seed, composite_chk, width, height, output, sampler_name="DDIM", iteration_count=1, strength=0.5, eta=0.1, ret_base64=False):

        if not type(input_image) is np.ndarray:
            if type(input_image) is str:
                input_image = read_image_to_np(input_image)
            if type(input_image) is Image.Image:
                input_image = np.array(input_image)

        if not type(mask_image) is np.ndarray:
            if type(mask_image) is str:
                mask_image = read_image_to_np(mask_image)
            if type(mask_image) is Image.Image:
                mask_image = np.array(mask_image)


        self.set_scheduler(sampler_name)
        if platform.system() == "Darwin":
            self.pipe = self.pipe.to("mps" if ia_check_versions.torch_mps_is_available else "cpu")
            self.pipe.enable_attention_slicing()
            torch_generator = torch.Generator(devices.cpu)
        else:
            # todo 这里需要注意，不做gc回收内存 需要使用.to("cuda")
            # self.pipe.enable_model_cpu_offload()
            self.pipe = self.pipe.to('cuda')
            if shared.xformers:
                ia_logging.info("Enable xformers memory efficient attention")
                self.pipe.enable_xformers_memory_efficient_attention()
            else:
                ia_logging.info("Enable attention slicing")
                self.pipe.enable_attention_slicing()
            torch_generator = torch.Generator("cuda")

        init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
        # mask_image = Image.fromarray(cv2.dilate(np.array(mask_image), np.ones((3, 3), dtype=np.uint8), iterations=4)).convert("L").filter(ImageFilter.GaussianBlur(3))
        # save_output_image_to_pil(mask_image, '/data1/aigc/phototrend/worker_data/inpaint_output')
        if width is None or height is None:
            width, height = init_image.size

        output_list = []
        iteration_count = iteration_count if iteration_count is not None else 1
        for count in range(int(iteration_count)):
            gc.collect()
            if seed < 0 or count > 0:
                seed = random.randint(0, 2147483647)

            generator = torch_generator.manual_seed(seed)
            # generator = torch.Generator(device="cpu").manual_seed(1)
            pipe_args_dict = {
                # "prompt": prompt,
                "image": init_image,
                "control_image": self.controlnet_image,
                "controlnet_conditioning_scale": self.controlnet_scale,
                "width": width,
                "height": height,
                "mask_image": mask_image,
                "num_inference_steps": ddim_steps,
                "guidance_scale": cfg_scale,
                # "negative_prompt": n_prompt,
                "generator": generator,
                "strength": strength,
                "eta": eta,
            }
            ia_logging.info(f"Pipe Args Dict:{pipe_args_dict}")
            if self.prompt_embeds is not None:
                pipe_args_dict['prompt_embeds'] = self.prompt_embeds
            else:
                pipe_args_dict['prompt'] = prompt
                ia_logging.info(f"prompt:{prompt}")
            if self.negative_prompt_embeds is not None:
                pipe_args_dict['negative_prompt_embeds'] = self.negative_prompt_embeds
            else:
                pipe_args_dict['negative_prompt'] = n_prompt
                ia_logging.info(f"negative_prompt:{n_prompt}")
            output_image = self.pipe(**pipe_args_dict).images[0]

            if composite_chk:
                dilate_mask_image = Image.fromarray(cv2.dilate(np.array(mask_image), np.ones((3, 3), dtype=np.uint8), iterations=4))
                output_image = Image.composite(output_image, init_image, dilate_mask_image.convert("L").filter(ImageFilter.GaussianBlur(3)))

            generation_params = {
                "Steps": ddim_steps,
                "Sampler": self.sampler_name,
                "CFG scale": cfg_scale,
                "Seed": seed,
                "Size": f"{width}x{height}",
                "Model": self.base_model,
            }

            generation_params_text = ", ".join([k if k == v else f"{k}: {v}" for k, v in generation_params.items() if v is not None])
            prompt_text = prompt if prompt else ""
            negative_prompt_text = "\nNegative prompt: " + n_prompt if n_prompt else ""
            infotext = f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()

            metadata = PngInfo()
            metadata.add_text("parameters", infotext)

            if output is not None:
                # img_idx = len(os.listdir(output))
                # save_name=os.path.join(output, f"{img_idx}.png")
                # output_image.save(save_name, pnginfo=metadata)
                save_output_image_to_pil(output_image, output)
                if ret_base64:
                    output_list.append(encode_to_base64(output_image))
                else:
                    output_list.append(output_image)
            else:
                if ret_base64:
                    output_list.append(encode_to_base64(output_image))
                else:
                    output_list.append(output_image)

        return output_list

