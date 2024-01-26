
import asyncio
import PIL.Image as Image
import copy
import os

from utils.pt_logging import ia_logging
from api.functions import queue,\
    encode_pil_to_base64,\
    generate_count, \
    G_PIPE, \
    interrogate, \
    batch_read_imgs_to_base64

from utils.req import async_back_host_generate


from api.pipe_tasks.base import Base


class CommodityPipe(Base):
    async def __call__(self, *args, **kwargs):
        ia_logging.info(f"{self.__class__.__name__} worker start: {self.params['id_task']}")
        await super().__call__(*args, **kwargs)
        ia_logging.info(f"{self.__class__.__name__} worker end: {self.params['id_task']}")

    async def action(self, **kwargs):
        input_image = Image.open(self.loca_img_path['input_image']).convert("RGB")
        mask = Image.open(self.loca_img_path['mask_image']).convert("RGB")
        batch_count = self.params['count']
        base_model = self.params['base_model']
        pos_prompt = self.params['prompt']
        neg_prompt = self.params['negative_prompt']
        sampler_name = self.params['sampler']
        strength = self.params['strength']
        width = self.params['width']
        height = self.params['height']
        steps = self.params['steps']
        cfg_scale = self.params['cfg_scale']

        c_count, d_count, host_list = generate_count(batch_count)
        sub_task = None
        if c_count > 0:
            dis_data = copy.deepcopy(self.base_params)
            dis_data['data']['input_images'] = [encode_pil_to_base64(input_image, True)]
            dis_data['data']['mask_image'] = encode_pil_to_base64(mask, True)
            dis_data['data']['preset'][0]['count'] = d_count
            batch_count = c_count

            print("host_list:", host_list)
            sub_task = asyncio.create_task(async_back_host_generate(host_list, dis_data, self.worker_dir_output))

        pos_prompt = pos_prompt % interrogate.interrogate(input_image) if '%s' in pos_prompt else pos_prompt
        self.pipe.set_controlnet_input(self.controlnet_sets)

        async def pipe_run_inpaint(input_image, mask, pos_prompt, neg_prompt, ddim_steps, cfg_scale,
                                   seed,
                                   composite_chk, sampler_name, batch_count, width, height,
                                   strength,
                                   eta, output, ret_base64):
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.pipe.run_inpaint, input_image, mask,
                                                pos_prompt,
                                                neg_prompt, ddim_steps, cfg_scale, seed,
                                                composite_chk, width,
                                                height, output, sampler_name, batch_count,
                                                strength,
                                                eta, ret_base64)
            return result

        await pipe_run_inpaint(
            input_image=input_image,
            mask=mask,
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            ddim_steps=steps,
            cfg_scale=cfg_scale,
            seed=-1,
            composite_chk=True,
            sampler_name=sampler_name,
            batch_count=batch_count,
            width=(int(width) // 8) * 8,
            height=(int(height) // 8) * 8,
            strength=strength,
            eta=31337,
            output=self.worker_dir_output,
            ret_base64=False
        )

        if sub_task is not None:
            # todo 判断子任务是否执行完成，如果未执行完，需重新加入队列
            await sub_task

        if not self.task_sync:
            res = await batch_read_imgs_to_base64([os.path.join(self.worker_dir_output, fname) for fname in os.listdir(self.worker_dir_output)])
            return res
        return []
