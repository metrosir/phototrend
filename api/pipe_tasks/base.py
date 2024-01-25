from abc import ABC, abstractmethod
from utils.pt_logging import ia_logging, log_echo
from scripts.inpaint import Inpainting
from utils.datadir import project_dir, dress_worker_history, api_generate_commodity_dir
from scripts.interrogate import InterrogateModels

import os
import pathlib
import json
import copy
import asyncio
import time
import PIL.Image as Image

from api.functions import queue,\
    saveimage, \
    G_PIPE, \
    lineart_image,\
    scribble_xdog, \
    interrogate


# 制作类型
gtype_commodity = 1
gtype_dress = 2


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

        return {
                "gtype":        params['gtype'] if 'gtype' in params else None,
                "id_task":      params['id_task'],
                "user_id":      params['user_id'] if 'user_id' in params else None,
                # 1：假模；2：真人
                "type":         params['data']['type'] if 'type' in params['data'] else None,

                # 主图
                "input_image":  params['data']['input_images'][0] if len(params['data']['input_images']) > 0 else None,
                # mask图
                "mask_image":   params['data']['mask_image'],

                # 参考图
                "reference_image": params['data']['reference_image'] if 'reference_image' in params['data'] else None,
                "reference": False if 'reference'not in params['data'] or params['data']['reference'] is None else params['data']['reference'],
                "reference_scale": 0.5 if 'reference_scale' not in params['data'] or params['data']['reference_scale'] is None else params['data']['reference_scale'],

                # 模特ID
                "face_id": None if 'face_id' not in params['data'] or params['data']['face_id'] is None else params['data']['face_id'],

                "base_model": params['data']['checkpoint_addr'],

                "count": params['data']['preset'][0]['count'],
                "seed": -1 if params['data']['preset'][0]['param']['seed'] is None else params['data']['preset'][0]['param']['seed'],
                "prompt": params['data']['preset'][0]['param']['prompt'],
                "negative_prompt": params['data']['preset'][0]['param']['negative_prompt'],
                "sampler": params['data']['preset'][0]['param']['sampler'],
                "width": params['data']['preset'][0]['param']['width'],
                "height": params['data']['preset'][0]['param']['height'],
                "steps":  10 if params['data']['preset'][0]['param']['steps'] is None else params['data']['preset'][0]['param']['steps'],
                "cfg_scale": 5 if 'cfg_scale' not in params['data']['preset'][0]['param'] or params['data']['preset'][0]['param']['cfg_scale'] is None else params['data']['preset'][0]['param']['cfg_scale'],
                "controlnets": params['data']['preset'][0]['param']['controlnets'],
                "strength": 0.5 if 'strength' not in params['data']['preset'][0]['param'] or params['data']['preset'][0]['param']['strength'] is None else params['data']['preset'][0]['param']['strength'],

                "callback_url": params['data']['callback_url'] if 'callback_url' in params['data'] else None,
        }


class Base(ABC, Params):
    data = []

    def __init__(self, base_params, pipe: [Inpainting, None], interr: [InterrogateModels, None], task_sync=True, gtype=gtype_commodity):

        self.base_params = base_params
        self.pipe = pipe

        self.interrogate = interr

        # if self.__class__.__name__ != RunWorker.__name__:
        self.params = self.clean_params(base_params) if base_params is not None else None

        self.controlnet_sets = []

        self.worker_dir_input = None
        self.worker_dir_output = None

        self.task_sync = task_sync

        self.gtype = gtype

        self.loca_img_path = {
            "reference_image": '',
            "input_image": '',
            "mask_image": '',
        }

    async def __call__(self, *args, **kwargs):
        try:
            self.before()
            # self.params = self.params_data()
            print('step 1')
            self.cut()
            print('step 2')
            self.data = await self.action()
            print('step 3')
            self.after()
            print('step 4')
        except Exception as e:
            log_echo(title="pipe error", msg=self.params, level="error", path=f"sync_tasks/{self.__class__.__name__}", exception=e, is_collect=True)
        finally:
            pass
        return self.data

    def before(self):
        log_echo(title="pipe before", msg={
            'params': copy.deepcopy(self.params)
        }, level="info", path=f"sync_tasks/{self.__class__.__name__}")

        print("self.gtype:", self.gtype)
        if self.gtype == gtype_dress:
            self.worker_dir_input = dress_worker_history.format(worker_id=self.params['id_task'], type='input')
            self.worker_dir_output = dress_worker_history.format(worker_id=self.params['id_task'], type='output')

        # todo 兼容商品图
        if self.gtype == gtype_commodity:
            self.worker_dir_input = api_generate_commodity_dir.format(id_task=self.params['id_task'], type='input')
            self.worker_dir_output = api_generate_commodity_dir.format(id_task=self.params['id_task'], type='output')

        if not os.path.exists(self.worker_dir_input):
            pathlib.Path.mkdir(pathlib.Path(self.worker_dir_input), parents=True, exist_ok=True)

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

        self.base_params['data']['input_images'][0] = '' if not self.base_params['data']['input_images'][0].startswith('http') else self.base_params['data']['input_images'][0]
        self.base_params['data']['mask_image'] = '' if not self.base_params['data']['mask_image'].startswith('http') else self.base_params['data']['mask_image']
        if 'reference_image' in self.base_params['data']:
            self.base_params['data']['reference_image'] = '' if not self.base_params['data']['reference_image'].startswith('http') else self.base_params['data']['reference_image']


    @abstractmethod
    async def action(self, **kwargs):
        pass

    def after(self):
        pass

    def cut(self):
        global G_PIPE
        def iscut():
            from utils.constant import contr_module_map_path
            controlnets = []
            try:
                is_cut = False
                if self.pipe is None:
                    self.pipe = G_PIPE
                    if self.pipe is None:
                        is_cut = True
                if not is_cut and self.params['base_model'] != self.pipe.base_model:
                    is_cut = True
                curr_model_paths = []
                old_model_paths = []
                for idx, contorlnet in enumerate(self.params['controlnets']):
                    if 'inpaint_only+lama' in contorlnet['controlnet_module']:
                        self.params['strength'] = contorlnet['weight']
                        continue

                    if contorlnet['controlnet_module'].find('ip-adapter') != -1:
                        continue

                    if contorlnet['controlnet_module'] not in contr_module_map_path:
                        raise Exception(
                            f"controlnet_module: {contorlnet['controlnet_module']} not in contr_module_map_path")

                    curr_model_paths.append(contr_module_map_path[contorlnet['controlnet_module']]['model_path'])
                    controlnets.append({
                        'low_cpu_mem_usage': False,
                        'model_path': contr_module_map_path[contorlnet['controlnet_module']]['model_path'],
                        'subfolder': contr_module_map_path[contorlnet['controlnet_module']]['subfolder'],
                        'device_map': None,
                    })
                    self.controlnet_sets.append({
                        'scale': contorlnet['weight'],
                        'image': self.controlnet_input_img(contorlnet['controlnet_module']),
                    })
                if not is_cut:
                    for idx, controlnet in enumerate(self.pipe.controlnets):
                        if controlnet is not None:
                            old_model_paths.append(controlnet['model_path'])

                    if set(curr_model_paths) != set(old_model_paths):
                        is_cut = True

                return is_cut, controlnets
            except Exception as e:
                raise e
        isc, controlnets = iscut()
        print("isc:", isc)
        if isc:
            try:
                self.pipe = G_PIPE = Inpainting(
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
            input_image = cv2.imread(self.loca_img_path['input_image'])
            pose_img = pose(input_image)
            openpose_img = Image.fromarray(pose_img)
            if is_save:
                openpose_img.save(os.path.join(self.worker_dir_input, 'openpose.png'))
            return openpose_img
        elif module.find('lineart') != -1:
            # from scripts.piplines.controlnet_pre import lineart_image
            lineart_img = lineart_image(input_image=self.loca_img_path['mask_image'], width=self.params['width'])
            if is_save:
                lineart_img.save(os.path.join(self.worker_dir_input, 'lineart.png'))
            return lineart_img
        elif module.find('scribble') != -1:
            scribble_img = scribble_xdog(img=self.loca_img_path['mask_image'], res=self.params['width'])
            if is_save:
                scribble_img.save(os.path.join(self.worker_dir_input, 'scribble.png'))
            return scribble_img
        raise Exception(f"controlnet_input_img module: {module} not in controlnet_input_img maps")


class InputWorkerData (Base):

    def __call__(self, *args, **kwargs):
        self.before()

    async def action(self, **kwargs):
        self.before()
        self.base_params['gtype'] = self.gtype
        queue.enqueue(json.dumps(self.base_params))


# queue_task_data = None


from api.pipe_tasks.commodity_pipe import CommodityPipe
from api.pipe_tasks.dress_pipe import DressPipe


class RunWorker(Base):

    async def __call__(self, *args, **kwargs):
        pass

    async def action(self, **kwargs):
        while 1:
            try:
                origin_data = queue.dequeue(is_complete=False)
                if origin_data is not None:
                    data = json.loads(origin_data)
                    self.__init__(data, None, None)
                    log_echo("Call Queue Task: ", msg={
                        "exec_args": json.dumps(data)
                    }, level='info', path='call_queue_task')
                    if self.params['gtype'] != gtype_dress:
                        pipe = CommodityPipe(data, G_PIPE, interrogate, gtype=gtype_commodity)
                        await pipe()
                    else:
                        pipe = DressPipe(data, G_PIPE, interrogate, gtype=gtype_dress)
                        await pipe()
                    queue.complete(origin_data)
            except Exception as e:
                log_echo("API Call Queue Error", msg={
                    "exec_args": json.dumps(origin_data)
                }, exception=e, is_collect=True, path='call_queue_task')
            finally:
                time.sleep(0.5)
