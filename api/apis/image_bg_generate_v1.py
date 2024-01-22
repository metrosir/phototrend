from api.base import ApiBase
from api.functions import *

from api.pipe_tasks.base import InputWorkerData, RunWorker, CommodityPipe


class ImageBgGenerateV1(ApiBase):

    def params_data(self):
        return self.params

    async def action(self):

        pipe = CommodityPipe(self.request, G_PIPE, interrogate, task_sync=False)
        return await pipe()

        data = self.request
        saveimage(
            id_task=data['id_task'],
            _type="input",
            images=[data['data']['input_images'][0], data['data']['mask_image']]
        )

        type_enum, input_image, \
            mask, base_model, pos_prompt, \
            neg_prompt, batch_count, sampler_name, contr_inp_weight, \
            contr_ipa_weight, contr_lin_weight, width, height, \
            contr_scribble_weight, steps, cfg_scale \
            = commodity_image_generate_api_params(data.get('data'))

        input_image = decode_base64_to_image(input_image)
        mask = decode_base64_to_image(mask)

        pos_prompt = pos_prompt % interrogate.interrogate(input_image) if '%s' in pos_prompt else pos_prompt
        pipe = set_model()

        lineart_input_img = \
            lineart_image(input_image=input_image, width=width)

        scribble_img = scribble_xdog(img=mask, res=width)
        saveimage(id_task=data['id_task'], _type="input", images=[lineart_input_img, scribble_img])
        pipe.set_controlnet_input([
            {
                'scale': contr_lin_weight,
                'image': lineart_input_img
            },
            {
                'scale': contr_scribble_weight,
                'image': scribble_img
            }
        ])

        async def pipe_run_inpaint():
            return pipe.run_inpaint(
                input_image=input_image,
                mask_image=mask,
                prompt=pos_prompt,
                n_prompt=neg_prompt,
                ddim_steps=steps,
                cfg_scale=cfg_scale,
                seed=-1,
                composite_chk=True,
                # sampler_name="UniPC",
                sampler_name=sampler_name,
                iteration_count=batch_count,
                width=(int(width) // 8) * 8,
                height=(int(height) // 8) * 8,
                strength=contr_inp_weight,
                eta=31337,
                output=datadir.api_generate_commodity_dir.format(id_task=data['id_task'], type="output", ),
                ret_base64=True
            )

        main_task = asyncio.create_task(pipe_run_inpaint())
        ret = await main_task
        return ret
