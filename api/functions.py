
from utils.image import decode_base64_to_image
from PIL import Image
from utils.pt_logging import log_echo

import os
import pathlib
import utils.datadir as datadir


def saveimage(id_task, _type: str, images: list, img_dir=None, file_names=[]):
    '''
    :param img_dir:
    :param file_names:
    :param id_task:
    :param _type: enumerate:input,output
    :param images:
    :return:
    '''
    from utils.constant import PT_ENV
    if PT_ENV is None:
        return None
    if img_dir is None:
        img_dir = datadir.api_generate_commodity_dir.format(id_task=id_task, type=_type)
    try:
        pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)
        idx = len(os.listdir(img_dir))
        ims_len = len(images)
        files_len = len(file_names)
        for f_idx, im in enumerate(images):
            if type(im) is not Image.Image:
                if len(file_names) > 0 and ims_len == files_len:
                    decode_base64_to_image(im).save(f"{img_dir}/{file_names[f_idx]}")
                else:
                    decode_base64_to_image(im).save(f"{img_dir}/{idx}.png")
            else:
                if len(file_names) > 0 and ims_len == files_len:
                    im.save(f"{img_dir}/{file_names[f_idx]}")
                else:
                    im.save(f"{img_dir}/{idx}.png")
            idx = idx + 1
    except Exception as e:
        log_echo("API Error", msg={"id_task": id_task}, exception=e, is_collect=True, )


from utils.constant import mode_params, self_innovate_mode, init_model, api_queue_dir, hosts
from .call_queue import LocalFileQueue as Queue
import json
import time
import copy
import asyncio
from utils.pt_logging import ia_logging, log_echo
from utils.image import convert_png_to_mask, mask_invert, remove_bg, decode_base64_to_image, encode_to_base64, \
    encode_pil_to_base64

from scripts.interrogate import InterrogateModels
from scripts.inpaint import Inpainting
from utils.cmd_args import opts as shared
from scripts.piplines.controlnet_pre import lineart_image, scribble_xdog

queue = Queue(api_queue_dir)
interrogate = InterrogateModels()

def generate_count(count: int):
    if count is None or count < 1:
        return count, 0, []
    from utils.constant import hosts
    import utils.utils as utils
    if not hosts or hosts is None:
        return count, 0, []
    hosts_list = hosts.split(",")
    if len(hosts_list) < 1:
        return count, 0

    local_ip = utils.get_local_ip()
    if hosts.find(local_ip) != -1:
        hosts_list.remove(local_ip)

    dis = count // (len(hosts_list) + 1)
    current_host = dis
    if dis * (len(hosts_list) + 1) != count:
        current_host = dis + 1
    return current_host, dis, hosts_list


def commodity_image_generate_api_params(request_data, id_task=None):
    if len(request_data['input_images']) < 1 or request_data['mask_image'] == '':
        if id_task is not None:
            input_image = Image.open(
                datadir.api_generate_commodity_dir.format(id_task=id_task, type="input", ) + '/0.png').convert("RGB")
            mask = Image.open(
                datadir.api_generate_commodity_dir.format(id_task=id_task, type="input", ) + '/1.png').convert("RGB")
    else:

        input_image = request_data['input_images'][0]
        mask = request_data['mask_image']

    # 1:商品图；2:模特图
    type_enum = 1 if 'type' not in request_data else int(request_data['type'])
    # for t in enumerate

    base_model = request_data['checkpoint_addr']

    pos_prompt = request_data['preset'][0]['param']['prompt']
    neg_prompt = request_data['preset'][0]['param']['negative_prompt']
    batch_count = request_data['preset'][0]['count']
    sampler_name = request_data['preset'][0]['param']['sampler']
    width = request_data['preset'][0]['param']['width']
    height = request_data['preset'][0]['param']['height']
    steps = request_data['preset'][0]['param']['steps']
    steps = 10
    cfg_scale = request_data['preset'][0]['param']['cfg_scale']
    if cfg_scale is None or cfg_scale == 0:
        cfg_scale = 7.5
    contr_inp_weight = mode_params[self_innovate_mode]['inpaint_weight']
    contr_ipa_weight = mode_params[self_innovate_mode]['ip-adapter_weight']
    contr_lin_weight = mode_params[self_innovate_mode]['lineart_weight']
    contr_scribble_weight = mode_params[self_innovate_mode]['scribble_weight']
    if len(request_data['preset'][0]['param']['controlnets']) > 0:
        for controlnet in request_data['preset'][0]['param']['controlnets']:
            if controlnet['controlnet_module'] == 'inpaint_only+lama':
                contr_inp_weight = controlnet['weight']
            elif controlnet['controlnet_module'] == 'ip-adapter_clip_sd15':
                contr_ipa_weight = controlnet['weight']
            elif controlnet['controlnet_module'] == 'lineart_realistic':
                contr_lin_weight = controlnet['weight']
            elif controlnet['controlnet_module'] == 'scribble_xdog':
                contr_scribble_weight = controlnet['weight']

    return type_enum, input_image, mask, base_model, pos_prompt, neg_prompt, batch_count, sampler_name, contr_inp_weight, contr_ipa_weight, contr_lin_weight, width, height, contr_scribble_weight, steps, cfg_scale


G_PIPE = None


def set_model():
    global G_PIPE
    if G_PIPE is not None:
        return G_PIPE
    G_PIPE = Inpainting(
        base_model=init_model['base_mode'],
        subfolder=None,
        controlnet=init_model['controlnets'],
        textual_inversion=init_model['textual_inversion'],
    )
    return G_PIPE


if shared.setup_mode:
    set_model()

queue_task_data = None

async def call_queue_task_sub_dress():
    pass


async def call_queue_task():
    global queue_task_data
    from utils.req import async_back_host_generate
    while 1:
        # queue_task_data = None
        try:
            origin_data = queue.dequeue(is_complete=False)
            if origin_data is not None:
                data = json.loads(origin_data)
                queue_task_data = data
                log_echo("Call Queue Task: ", msg={
                    "exec_args": json.dumps(data)
                }, level='info', path='call_queue_task')

                type_enum, input_image, mask, base_model, pos_prompt, neg_prompt, batch_count, sampler_name, contr_inp_weight, contr_ipa_weight, contr_lin_weight, width, height, contr_scribble_weight, steps, cfg_scale \
                    = commodity_image_generate_api_params(data['data'], id_task=data['id_task'])
                if type_enum == 1:
                    if type(input_image) is str:
                        input_image = decode_base64_to_image(input_image)
                    if type(mask) is str:
                        mask = decode_base64_to_image(mask)

                    ia_logging.info(f"Call task:{data['id_task']}, data:{data}")
                    c_count, d_count, host_list = generate_count(batch_count)
                    sub_task = None
                    if c_count > 0:
                        dis_data = copy.deepcopy(data)
                        dis_data['data']['input_images'] = [
                            encode_pil_to_base64(input_image, True),
                        ]
                        dis_data['data']['mask_image'] = encode_pil_to_base64(mask, True)
                        dis_data['data']['preset'][0]['count'] = d_count
                        batch_count = c_count

                        # async def async_back_host_generate(host_list, dis_data, output_dir):
                        #     # todo
                        #     # return await back_host_generate(host_list, dis_data, output_dir=output_dir)
                        #     return await back_host_generate(host_list, dis_data, output_dir=output_dir)
                        print("host_list:", host_list)
                        sub_task = asyncio.create_task(async_back_host_generate(host_list, dis_data,
                                                                                datadir.api_generate_commodity_dir.format(
                                                                                    id_task=data['id_task'],
                                                                                    type="output", )))

                    pos_prompt = pos_prompt % interrogate.interrogate(input_image) if '%s' in pos_prompt else pos_prompt

                    if G_PIPE is None:
                        pipe = set_model()
                    else:
                        pipe = G_PIPE

                    lineart_input_img = lineart_image(input_image=input_image, width=width)
                    # lineart_mask_img = lineart_image(input_image=mask, width=width)
                    scribble_img = scribble_xdog(img=mask, res=width)
                    saveimage(id_task=data['id_task'], _type="input", images=[lineart_input_img, scribble_img])
                    pipe.set_controlnet_input([
                        # {
                        #     'scale': contr_ipa_weight,
                        #     'image': input_image,
                        # },
                        {
                            'scale': contr_lin_weight,
                            'image': lineart_input_img
                        },
                        # {
                        #     'scale': 0.55,
                        #     'image': lineart_mask_img
                        # }
                        {
                            'scale': contr_scribble_weight,
                            'image': scribble_img
                        }
                    ])

                    async def pipe_run_inpaint(input_image, mask, pos_prompt, neg_prompt, ddim_steps, cfg_scale, seed,
                                               composite_chk, sampler_name, batch_count, width, height, contr_inp_weight,
                                               eta, output, ret_base64):
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(None, pipe.run_inpaint, input_image, mask, pos_prompt,
                                                            neg_prompt, ddim_steps, cfg_scale, seed, composite_chk, width,
                                                            height, output, sampler_name, batch_count, contr_inp_weight,
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
                        contr_inp_weight=contr_inp_weight,
                        eta=31337,
                        output=datadir.api_generate_commodity_dir.format(id_task=data['id_task'], type="output", ),
                        ret_base64=True
                    )
                    if sub_task is not None:
                        # todo 判断子任务是否执行完成，如果未执行完，需重新加入队列
                        sub_task_result = await sub_task
                        print("sub_task_result:", sub_task_result)
                else:
                    await call_queue_task_sub_dress()
                queue.complete(origin_data)
        except Exception as e:
            log_echo("API Call Queue Error", msg={
                "exec_args": json.dumps(data)
            }, exception=e, is_collect=True, path='call_queue_task')
        finally:
            time.sleep(0.5)


async def batch_read_imgs_to_base64(img_list):
    async def encode_image_to_base64(path):
        def sync_encode_image_to_base64(path):
            with Image.open(path) as im:
                if im.size == (0, 0):
                    return None
                im = im.convert("RGB")
                return encode_to_base64(im)
        return await asyncio.to_thread(sync_encode_image_to_base64, path)

    async def cys():
        tasks = [encode_image_to_base64(path) for path in img_list]
        res = await asyncio.gather(*tasks)
        return res

    return await cys()
