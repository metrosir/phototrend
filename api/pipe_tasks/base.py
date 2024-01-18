from abc import ABC, abstractmethod
from utils.pt_logging import ia_logging, log_echo
from scripts.inpaint import Inpainting
from utils.datadir import project_dir, dress_worker_input_history, dress_worker_output_history
from scripts.interrogate import InterrogateModels

import os
import pathlib

from api.functions import saveimage


class Params:

    params = None

    def clean_params(self, params):
        if 'data' not in params:
            raise Exception("params error: data field is empty")
        if 'preset' not in params['data'] or len(params['data']['preset']) < 1:
            raise Exception("params error: preset field is empty")
        if 'param' not in params['data']['preset'][0]:
            raise Exception("params error: param field is empty")
        if 'controlnets' not in params['data']['preset'][0]['param'] or not isinstance(params['data']['preset'][0]['param']['controlnets'], list) or len(params['data']['preset'][0]['param']['controlnets']) < 1:
            raise Exception("params error: controlnets field is empty")

        self.params = {
                "id_task":      params['id_task'],
                "user_id":      params['user_id'] if 'user_id' in params else None,

                "input_image":  params['data']['input_images'][0] if len(params['data']['input_images']) > 0 else None,
                "mask_image":   params['data']['mask_image'],

                # 参考图
                "reference_image": params['data']['reference_image'] if 'reference_image' in params['data'] else None,
                "base_model": params['data']['checkpoint_addr'],

                "count": params['data']['preset'][0]['count'],
                "seed":  params['data']['preset'][0]['param']['seed'],
                "prompt": params['data']['preset'][0]['param']['prompt'],
                "negative_prompt": params['data']['preset'][0]['param']['negative_prompt'],
                "sampler": params['data']['preset'][0]['param']['sampler'],
                "width": params['data']['preset'][0]['param']['width'],
                "height": params['data']['preset'][0]['param']['height'],
                "steps": params['data']['preset'][0]['param']['steps'],
                "cfg_scale": params['data']['preset'][0]['param']['cfg_scale'],
                "controlnets": params['data']['preset'][0]['param']['controlnets'],
                "strength": 0.5,

                "callback_url": params['data']['checkpoint_addr'],
        }


class Base(ABC, Params):
    def __init__(self, base_params, pipe: [Inpainting, None], interrogate: [InterrogateModels, None]):
        self.base_params = base_params
        self.pipe = pipe
        self.interrogate = interrogate

        self.params = self.clean_params(base_params)

        self.controlnet_sets = []

        self.worker_dir_input = None
        self.worker_dir_output = None

        self.loca_img_path = {
            "reference_image": '',
            "input_image": '',
            "mask_image": '',
        }

    async def __call__(self, *args, **kwargs):
        try:
            self.before()
            # self.params = self.params_data()
            self.cut()
            self.data = await self.action()
            self.after()
        except Exception as e:
            log_echo(title="pipe error", msg=self.params, level="error", path=f"sync_tasks/{self.__class__.__name__}", exception=e)
        finally:
            pass

    def before(self):
        log_echo(title="pipe before", msg={
            'params': self.params
        }, level="info", path=f"sync_tasks/{self.__class__.__name__}")

        self.worker_dir_input = dress_worker_input_history.format(worker_id=self.params['id_task'])
        if not os.path.exists(self.worker_dir_input):
            pathlib.Path.mkdir(pathlib.Path(self.worker_dir_input), parents=True, exist_ok=True)

        self.worker_dir_output = dress_worker_output_history.format(worker_id=self.params['id_task'])
        if not os.path.exists(self.worker_dir_output):
            pathlib.Path.mkdir(pathlib.Path(self.worker_dir_output), parents=True, exist_ok=True)

        loca_file_tags = list(self.loca_img_path.keys())
        files = os.listdir(self.worker_dir_input)
        if len(files) > 0:
            for file in files:
                im_path = os.path.join(self.worker_dir_input, file)
                for lf in loca_file_tags:
                    if file.startswith(f'{lf}.png'):
                        self.loca_img_path[lf] = im_path

        def check_save_img(fields: list):
            for field in fields:
                if self.loca_img_path[field] == '' and self.params[field] is not None:
                    path = os.path.join(self.worker_dir_input, f'{field}.png')
                    if self.params[field].startswith('http'):
                        import wget
                        wget.download(self.params[field], path)
                    else:
                        saveimage(
                            id_task=self.params['id_task'],
                            _type='input',
                            images=[self.params[field]],
                            img_dir=self.worker_dir_input,
                            file_names=[f'{field}.png']
                        )
                    self.loca_img_path[field] = path
        check_save_img(loca_file_tags)

        if self.loca_img_path['input_image'] == '':
            raise Exception("input_image is None")


    @abstractmethod
    async def action(self):
        pass

    def after(self):
        pass

    def cut(self):
        def iscut():
            from utils.constant import contr_module_map_path
            controlnets = []
            try:
                is_cut = False
                if self.params['base_model'] != self.pipe.base_model:
                    is_cut = True
                if len(self.pipe.controlnets) != len(self.params['controlnets']):
                    is_cut = True
                for idx, contorlnet in enumerate(self.params['controlnets']):
                    if 'inpaint_only+lama' in contorlnet['controlnet_module']:
                        self.params['strength'] = contorlnet['weight']
                        continue
                    if contorlnet['controlnet_module'] not in contr_module_map_path:
                        raise Exception(
                            f"controlnet_module: {contorlnet['controlnet_module']} not in contr_module_map_path")
                    if self.pipe.controlnets[idx]['model_path'] != contr_module_map_path[contorlnet['controlnet_module']]:
                        is_cut = True
                    controlnets.append({
                        'low_cpu_mem_usage': False,
                        'model_path': contr_module_map_path[contorlnet['controlnet_module']]['model_path'],
                        'subfolder': contr_module_map_path[contorlnet['controlnet_module']]['subfolder'],
                        'device_map': None,
                    })
                    self.controlnet_sets.append({
                        'scale': contorlnet['scale'],
                        'image': self.controlnet_input_img(contorlnet['controlnet_module']),
                    })
                return is_cut, controlnets
            except Exception as e:
                raise e
        isc, controlnets = iscut()
        if isc:
            try:
                self.pipe = self.pipe.__init__(
                    base_model=self.params['base_model'],
                    subfolder=None,
                    controlnet=controlnets,
                    textual_inversion={},
                )
            except Exception as e:
                raise e

    def controlnet_input_img(self, module: str, is_save: bool = True):

        if module.find('pose') != -1:
            from annotator.dwpose import DWposeDetector
            import cv2
            import PIL.Image as Image
            pose = DWposeDetector(
                f"{project_dir}/repositories/DWPose/ControlNet-v1-1-nightly/annotator/ckpts/yolox_l.onnx",
                f"{project_dir}/repositories/DWPose/ControlNet-v1-1-nightly/annotator/ckpts/dw-ll_ucoco_384.onnx")
            input_image = cv2.imread(self.params['input_image'])
            pose_img = pose(input_image)
            openpose_img = Image.fromarray(pose_img)
            if is_save:
                openpose_img.save(os.path.join(self.worker_dir_input, 'openpose.png'))
            return openpose_img
        elif module.find('lineart') != -1:
            from scripts.piplines.controlnet_pre import lineart_image
            lineart_img = lineart_image(input_image=self.params['mask_image'], width=self.params['width'])
            if is_save:
                lineart_img.save(os.path.join(self.worker_dir_input, 'lineart.png'))
            return lineart_img
        raise Exception(f"controlnet_input_img module: {module} not in controlnet_input_img maps")
