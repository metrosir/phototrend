from api.pipe_tasks.base import Base
from scripts.inpaint import Inpainting

from annotator.dwpose import DWposeDetector
from PIL import Image
from scripts.piplines.controlnet_pre import lineart_image, scribble_xdog, canny

from utils.datadir import project_dir

import cv2


class DressPipe(Base):

    def params_data(self):
        return {}

    async def action(self):
        input_image = cv2.imread(self.loca_img_path['input_image'])
        print("self.loca_img_path['input_image']:", self.loca_img_path['input_image'])
        self.params['prompt'] = \
            self.params['prompt'] % self.interrogate.interrogate(Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))) if '%s' in self.params['prompt'] else self.params['prompt']

        self.pipe.load_textual_inversion(
            [f'{project_dir}/models/textual_inversion/negative_prompt/epiCPhotoGasm-colorfulPhoto-neg.pt',
             f'{project_dir}/models/textual_inversion/negative_prompt/epiCPhotoGasm-softPhoto-neg.pt'],
        )
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
            width=(int(self.params['width']) // 8) * 8,
            height=(int(self.params['height']) // 8) * 8,
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
        )
        controlnet_set_data = []
        # for contl in self.params['controlnets']:
        # self.pipe.set_controlnet_input()
