import time

from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from typing import Optional
import os
import pathlib
from utils.image import convert_png_to_mask, mask_invert,remove_bg, decode_base64_to_image
import utils.datadir as datadir
# from utils.req import interrogate
from utils.utils import project_dir
from PIL import Image
import scripts.interrogate
# from .models import *

from utils.constant import mode_params, self_innovate_mode

interrogate = scripts.interrogate.InterrogateModels()


def commodity_image_generate_api_params(request_data):
    input_image = request_data['input_images'][0]
    mask = request_data['mask_image']
    base_model = request_data['checkpoint_addr']
    pos_prompt = request_data['preset'][0]['param']['prompt']
    neg_prompt = request_data['preset'][0]['param']['negative_prompt']
    batch_count = request_data['preset'][0]['count']
    sampler_name = request_data['preset'][0]['param']['sampler']
    width = request_data['preset'][0]['param']['width']
    height = request_data['preset'][0]['param']['height']
    contr_inp_weight = mode_params[self_innovate_mode]['inpaint_weight']
    contr_ipa_weight = mode_params[self_innovate_mode]['ip-adapter_weight']
    contr_lin_weight = mode_params[self_innovate_mode]['lineart_weight']
    if len(request_data['preset'][0]['param']['controlnets']) > 0:
        for controlnet in request_data['preset'][0]['param']['controlnets']:
            if controlnet['controlnet_module'] == 'inpaint_only+lama':
                contr_inp_weight = controlnet['weight']
            elif controlnet['controlnet_module'] == 'ip-adapter_clip_sd15':
                contr_ipa_weight = controlnet['weight']
            elif controlnet['controlnet_module'] == 'lineart_realistic':
                contr_lin_weight = controlnet['weight']
    return input_image, mask, base_model, pos_prompt, neg_prompt, batch_count, sampler_name, contr_inp_weight, contr_ipa_weight, contr_lin_weight, width, height


class Api:
    def __init__(self, app: FastAPI):
        self.app = app
        self.app.add_api_route("/iframe", self.read_html_file, methods=["get"], response_class=HTMLResponse)
        self.app.add_api_route("/iframe_clothes", self.read_clothes_html_file, methods=["get"], response_class=HTMLResponse)
        self.app.add_api_route("/upload_image", self.upload_image, methods=["post"])
        self.app.add_api_route("/upload_clothes_image", self.upload_clothes_image, methods=["post"])
        self.app.add_api_route("/deft_scene", self.deft_scene, methods=["get"])
        self.app.add_api_route("/human_imag", self.human_imag, methods=["get"])
        self.app.add_api_route("/clothes_imag", self.clothes_imag, methods=["get"])

        self.app.add_api_route("/v1/image/interrogate", self.interrogate, methods=["post"])

        self.app.add_api_route("/v1/commodity_image/generate", self.commodity_image_generate, methods=["post"])

    def interrogate(self):
        pass

    async def commodity_image_generate(self, request: Request):
        data = await request.json()
        input_image, mask, base_model, pos_prompt, neg_prompt, batch_count, sampler_name, contr_inp_weight, contr_ipa_weight, contr_lin_weight, width, height \
            = commodity_image_generate_api_params(data.get('data'))
        from scripts.inpaint import Inpainting
        from scripts.piplines.controlnet_pre import lineart_image
        strt_time = time.time()
        input_image = decode_base64_to_image(input_image)
        mask = decode_base64_to_image(mask)

        ip = Inpainting(
            base_model=base_model,
            subfolder=None,
            controlnet=[
                {
                    'low_cpu_mem_usage': False,
                    'device_map': None,
                    'model_path': 'metrosir/phototrend',
                    "subfolder": 'controlnets/ip-adapter-plus',
                    'scale': contr_ipa_weight,
                    'image': input_image,
                    'local_files_only': False
                },
                {
                    'low_cpu_mem_usage': False,
                    'model_path': 'metrosir/phototrend',
                    'subfolder': 'controlnets/lineart-fp16',
                    'scale': contr_lin_weight,
                    'device_map': None,
                    'image': lineart_image(input_image=input_image, width=width)
                }
            ]
        )

        ip.set_textual_inversion(
            f'{project_dir}/models/textual_inversion/negative_prompt/realisticvision-negative-embedding.pt',
            'realisticvision-negative-embedding',
            'string_to_param',
        )
        ret = ip.run_inpaint(
            input_image=input_image,
            mask_image=mask,
            prompt=pos_prompt,
            n_prompt=neg_prompt,
            ddim_steps=30,
            cfg_scale=7.5,
            seed=-1,
            composite_chk=True,
            # sampler_name="UniPC",
            sampler_name=sampler_name,
            iteration_count=batch_count,
            width=(int(width) // 8) * 8,
            height=(int(height) // 8) * 8,
            strength=contr_inp_weight,
            eta=31337,
            output=None
        )
        end_time = time.time()
        duration = end_time - strt_time
        print(f"commodity_image_generate_api_params time:{end_time - strt_time}")
        return {"data": ret, "duration": duration}

    def read_html_file(self):
        file_path = f'{project_dir}/view/editimg.html'
        with open(file_path, "r", encoding='utf-8') as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)

    def read_clothes_html_file(self):
        file_path = f'{project_dir}/view/clothes_editimg.html'
        with open(file_path, "r", encoding='utf-8') as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)

    def upload_image(self, file: UploadFile = File(...), img_type: Optional[int] = None):
        if not os.path.exists(datadir.commodity_merge_scene_image_dir):
            pathlib.Path(datadir.commodity_merge_scene_image_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(datadir.merge_after_mask_image_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(datadir.mask_image_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(datadir.merge_after_mask_cut_image_dir).mkdir(parents=True, exist_ok=True)

        # 制作mask
        if img_type == 2:
            try:
                contents = file.file.read()
                after_mask_path = f'{datadir.merge_after_mask_image_dir}/{datadir.get_file_idx()}.png'
                mask_path = f'{datadir.mask_image_dir}/{datadir.get_file_idx()}.png'
                merge_after_mask_cut_image_dir = f'{datadir.merge_after_mask_cut_image_dir}/{datadir.get_file_idx()}.png'
                with open(after_mask_path, 'wb') as f:
                    f.write(contents)

                convert_png_to_mask(after_mask_path, mask_path)
                mask_invert(mask_path, merge_after_mask_cut_image_dir)
                # remove_bg(after_mask_path, mask_path, True, False)
                # mask_invert(mask_path, after_mask_path)

                # convert_png_to_mask(mask_path, after_mask_path)
            except Exception as e:
                error_message = str(e)
                return {"data": f"{mask_path}, type:{img_type}, error:{error_message}"}
            return {"data": f"{mask_path}, type:{img_type}"}
        # 场景图
        else:
            i_path = f'{datadir.commodity_merge_scene_image_dir}/{datadir.get_file_idx()}.png'
            contents = file.file.read()
            with open(i_path, 'wb') as f:
                f.write(contents)
            # return {"data": f"{i_path}, type{img_type}", "caption": interrogate(i_path)}
            img = Image.open(i_path)
            img = img.convert('RGB')

            return {"data": f"{i_path}, type{img_type}","caption": interrogate.interrogate(img)}


    def upload_clothes_image(self, file: UploadFile = File(...), img_type: Optional[int] = None):
        if not os.path.exists(datadir.clothes_merge_scene_dir):
            pathlib.Path(datadir.clothes_merge_scene_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(datadir.clothes_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(datadir.clothes_mask_dir).mkdir(parents=True, exist_ok=True)

        # 制作mask
        if img_type == 2:
            try:
                contents = file.file.read()
                clothes_path = f'{datadir.clothes_dir}/{datadir.get_file_idx(check_dir=datadir.clothes_dir)}.png'
                mask_path = f'{datadir.clothes_mask_dir}/{datadir.get_file_idx(check_dir=datadir.clothes_dir)}.png'
                with open(clothes_path, 'wb') as f:
                    f.write(contents)

                convert_png_to_mask(clothes_path, mask_path)
            except Exception as e:
                error_message = str(e)
                return {"data": f"{mask_path}, type:{img_type}, error:{error_message}"}
            return {"data": f"{mask_path}, type:{img_type}"}
        # 场景图
        else:
            i_path = f'{datadir.clothes_merge_scene_dir}/{datadir.get_file_idx(check_dir=datadir.clothes_dir)}.png'
            contents = file.file.read()
            with open(i_path, 'wb') as f:
                f.write(contents)
            return {"data": f"{i_path}, type{img_type}", "caption": interrogate(i_path)}



    def deft_scene(self, type: Optional[int] = None):
        try:
            if type == 1:
                return FileResponse(f"{project_dir}/worker_data/template/768x1024.png")
            else:
                return FileResponse(f"{project_dir}/worker_data/template/800x1422.jpeg")
        except Exception as e:
            return {"message": f"There was an error reading the image:{str(e)}"}


    def human_imag(self):
        try:
            return FileResponse(f"{project_dir}/worker_data/template/human_image.png")
        except Exception as e:
            return {"message": f"There was an error reading the image:{str(e)}"}


    def clothes_imag(self):
        try:
            return FileResponse(f"{project_dir}/worker_data/template/clothes_image.png")
        except Exception as e:
            return {"message": f"There was an error reading the image:{str(e)}"}
