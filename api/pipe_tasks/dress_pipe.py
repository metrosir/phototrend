import sys

from api.pipe_tasks.base import Base
from scripts.inpaint import Inpainting
from typing import Any

from annotator.dwpose import DWposeDetector
from PIL import Image
from scripts.piplines.controlnet_pre import lineart_image, scribble_xdog, canny
from utils.pt_logging import ia_logging

from utils.datadir import project_dir

import cv2


class DressPipe(Base):

    async def __call__(self, *args, **kwargs):
        ia_logging.info(f"{self.__class__.__name__} worker start: {self.params['id_task']}")
        await super().__call__(*args, **kwargs)
        ia_logging.info(f"{self.__class__.__name__} worker end: {self.params['id_task']}")

    def params_data(self):
        return {}

    async def action(self, **kwargs):
        def callback(params):
            req_params = {
                "id_task": params['id_task'],
                "unupload":1,
                "app_id": "aispark",
                "user_id": params['uid'],
                "biz_call_back": params['call_back_url'],
                "data": [
                    {
                        # completed/fail/running
                        "status": params['status'],
                        "preview": [
                            {
                                "code": 0,
                                "total_num": int(params['count']),
                                "width": int(params['width']),
                                "height": int(params['height']),
                                "list": [
                                ],
                                "interrogate": {
                                    "clip_prompt": "",
                                    "tagger_prompts": None
                                }
                            }
                        ]
                    }
                ]
            }
            from utils.s3 import upload as s3_upload
            import uuid
            from utils.pt_logging import log_echo
            import inspect
            _, _, _, args = inspect.getargvalues(inspect.currentframe())

            if params['uid'] is None:
                params['uid'] = uuid.uuid4().hex[:8]
            try:
                image_url = s3_upload(params['image_path'], params['uid'], params['id_task'])
            except Exception as e:
                log_echo("dress pipe callback error", msg={
                    "params": args,
                }, exception=e, is_collect=True, path='call_queue_task')
                return

            from utils.req import async_req_base,requestbase
            try:
                req_params['data'][0]['preview'][0]['list'].append(image_url)
                req_data = requestbase(url=params['call_back_url'], method="post", data=req_params, headers={
                    'Content-Type': 'application/json',
                })
            except Exception as e:
                log_echo("dress pipe callback error", msg={
                    "params": args,
                }, exception=e, is_collect=True, path='call_queue_task')
                return

        input_image = cv2.imread(self.loca_img_path['input_image'])
        self.params['prompt'] = \
            self.params['prompt'] % self.interrogate.interrogate(Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))) if '%s' in self.params['prompt'] else self.params['prompt']

        self.pipe.load_textual_inversion(
            [f'{project_dir}/models/textual_inversion/negative_prompt/epiCPhotoGasm-colorfulPhoto-neg.pt',
             f'{project_dir}/models/textual_inversion/negative_prompt/epiCPhotoGasm-softPhoto-neg.pt'],
        )
        width = int((int(self.params['width']) // 8) * 8)
        height = int((int(self.params['height']) // 8) * 8)
        self.pipe.set_controlnet_input(self.controlnet_sets)
        self.pipe.run_inpaint(
            input_image=self.loca_img_path['input_image'],
            mask_image=self.loca_img_path['mask_image'],
            prompt=self.params['prompt'],
            n_prompt=self.params['negative_prompt'],
            ddim_steps=self.params['steps'],
            cfg_scale=self.params['cfg_scale'],
            seed=int(self.params['seed']),
            composite_chk=True,
            sampler_name=self.params['sampler'],
            iteration_count=self.params['count'],
            width=width,
            height=height,
            strength=self.params['strength'],
            eta=1.0,
            output=self.worker_dir_output,
            open_after=None,
            after_params=None,
            res_img_info=True,
            use_ip_adapter=self.params['reference'],
            # ipadapter_img=Image.open(self.loca_img_path['reference_image']).convert('RGB') if self.params['reference'] else None,
            ipadapter_img=Image.open(self.loca_img_path['reference_image']).convert('RGB'),
            ip_adapter_scale=self.params['reference_scale'],
            call_back_func=callback,
            call_back_func_params={
                "image_path": self.worker_dir_output + '{idx}.png',
                "call_back_url": self.params['callback_url'],
                "uid": self.params['user_id'],
                "id_task": self.params['id_task'],
                "count": self.params['count'],
                "status": "running",
                "width": width,
                "height": height,
            },
            # 二次绘制
            twice=True,
            twice_params={
                "strength": 0.4,
                "guess_mode": True,
                "num_inference_steps": 30,
            }
        )
        controlnet_set_data = []
        # for contl in self.params['controlnets']:
        # self.pipe.set_controlnet_input()
