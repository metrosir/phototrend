import json
import sys
import gradio as gr
import pathlib
import uuid
import os
import glob
import utils.datadir as datadir
from utils.req import generate_image, prompt, negative_prompt, negative_prompt_clothes, sd_models, sd_vae
# from utils.image import remove_bg, convert_png_to_mask
import utils.image as image_utils
from utils.cmd_args import opts as cmd_opts
import utils.constant as constant
from utils.utils import project_dir
import scripts.templatemanager as tm
import pandas as pd
import datetime

import api.api as Api
import time


def rmbg(image, is_result=False) -> str:
    if not os.path.exists(path=datadir.commodity_image_dir):
        pathlib.Path(datadir.commodity_image_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(datadir.commodity_rembg_image_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(datadir.commodity_rembg_mask_image_dir).mkdir(parents=True, exist_ok=True)

    commodity_image_path = f'{datadir.commodity_image_dir}/{datadir.get_file_idx(True)}.png'
    commodity_rembg_image_path = f'{datadir.commodity_rembg_image_dir}/{datadir.get_file_idx(True)}.png'
    commodity_rembg_mask_image_path = f'{datadir.commodity_rembg_mask_image_dir}/{datadir.get_file_idx(True)}.png'

    if is_result:
        image.save(commodity_image_path)
        image.save(commodity_rembg_image_path)
        return [commodity_rembg_image_path]
    else:
        image.save(commodity_image_path)
        image_utils.remove_bg(commodity_image_path, commodity_rembg_image_path)

        return [commodity_rembg_image_path]
        # remove_bg(commodity_rembg_image_path, commodity_rembg_mask_image_path, mask=True, alpha_matting=True)
    # convert_png_to_mask(commodity_rembg_image_path, commodity_rembg_mask_image_path)
    # ! rembg i -a $commodity_image_path $commodity_rembg_image_path
    # ! rembg i -a -om $commodity_image_path $commodity_rembg_mask_image_path
    # print("path:", [[commodity_rembg_image_path]])



def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths


def dir_uuid():
    return datadir.uuid


def refresh_history_img(type=1):
    if int(type)==1:
        dirs = datadir.generate_glob_img
    elif int(type)==3:
        dirs = datadir.generate_self_innovate_glob_img
    else:
        dirs = datadir.clothes_generate_glob_img
    return glob.glob(dirs)


def upload_rem_img_result(img):
    # return rmbg(img, True)
    img = img.crop(img.getbbox())
    path = rmbg(img, True)
    # print(f"path:{path}")
    # return  path
    # output_images.update(value=path)
    return gr.Gallery.update(value=path, visible=True)
    # return gr.Gallery.update(value=path[0], visible=True)


def generate(mode, select_model, select_vae, pos_prompt, neg_prompt, batch_count,
                                       contr_inp_weight, contr_ipa_weight, contr_lin_weight, generate_type, width, height, contr_scribble_weight, ddim_steps, sampler_name):
    if mode not in constant.generate_mode:
        raise Exception("mode not in constant.generate_mode")
    if mode == constant.sd_mode:
        return generate_image(select_model, select_vae, pos_prompt, neg_prompt, batch_count,
                                       contr_inp_weight, contr_ipa_weight, contr_lin_weight, generate_type, width, height)
    else:
        from scripts.inpaint import Inpainting
        from scripts.piplines.controlnet_pre import lineart_image,scribble_xdog
        idx = datadir.get_file_idx(check_dir=datadir.commodity_merge_scene_image_dir)
        generate_image_sub_dir = datadir.generate_self_innovate_image_dir.format(uuid=datadir.uuid, idx=idx)
        if not pathlib.Path(generate_image_sub_dir).exists():
            pathlib.Path(generate_image_sub_dir).mkdir(parents=True, exist_ok=True)
        comm_merge_scene_im = f'{datadir.commodity_merge_scene_image_dir}/{datadir.get_file_idx()}.png'
        mask_im = f'{datadir.merge_after_mask_cut_image_dir}/{datadir.get_file_idx()}.png'

        def save_image(images):
            dir = datadir.controlnet_images.format(uuid=datadir.uuid, idx=datadir.get_file_idx())
            if not pathlib.Path(dir).exists():
                pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
            file_list =os.listdir(dir)
            for images_idx, image in enumerate(images):
                # np 格式图像存储
                image.save(f'{dir}/{len(file_list)+images_idx}.png')

        if Api.gpipe is None:
            Api.gpipe = Api.set_model()

        from PIL import Image

        # scene_im = Image.open(comm_merge_scene_im)
        # scene_im = scene_im.convert("RGB")
        lineart_img = lineart_image(input_image=comm_merge_scene_im, width=width)
        scribble_img = scribble_xdog(img=mask_im, res=width)
        save_image([lineart_img, scribble_img])
        Api.gpipe.set_controlnet_input(
            [
                # {
                #     'scale': contr_ipa_weight,
                #     'image': scene_im,
                # },
                {
                    'scale': contr_lin_weight,
                    'image': lineart_img
                },
                {
                    'scale': contr_scribble_weight,
                    'image': scribble_img
                }
            ]
        )

        # ip_adapter
        # image_encoder_path = '/data1/aigc/phototrend/models/ip_adapter/image_encoder'
        # # ip_ckpt = '/data/aigc/stable-diffusion-webui/extensions/sd-webui-controlnet/models/ip-adapter_sd15_plus.pth'
        # ip_ckpt = '/data/aigc/stable-diffusion-webui/extensions/sd-webui-controlnet/models/ip-adapter-plus/diffusion_pytorch_model.bin'
        # device = 'cuda'
        # Api.gpipe.set_ip_adapter(image_encoder_path, ip_ckpt, device, num_tokens=16)
        # Api.gpipe.input_ip_adapter_condition(pil_image=image_utils.open_image_to_pil(comm_merge_scene_im), prompt=pos_prompt,
        #                                 negative_prompt=neg_prompt, scale=contr_ipa_weight)

        return Api.gpipe.run_inpaint(
            input_image=comm_merge_scene_im,
            mask_image=mask_im,
            prompt=pos_prompt,
            n_prompt=neg_prompt,
            ddim_steps=ddim_steps,
            cfg_scale=8.5,
            seed=-1,
            composite_chk=True,
            # sampler_name="Euler a",
            sampler_name=sampler_name,
            iteration_count=batch_count,
            width=(int(width)//8)*8,
            height=(int(height)//8)*8,
            strength=contr_inp_weight,
            eta=31337,
            output=generate_image_sub_dir
        )


def save_commdity_tmpe(template_name, template_img, template_size, shape, coordinate, type, pos_prompt, neg_prompt,contr_inp_weight, contr_lin_weight, width, height, contr_scribble_weight, ddim_steps, sampler_name):
    inf_params = {
        'prompt': pos_prompt,
        'neg_prompt': neg_prompt,
        'ctrl': {
            'inpaint': contr_inp_weight,
            'scribble': contr_scribble_weight,
            'lineart': contr_lin_weight,
        },
        'width': width,
        'height': height,
        'sampler_steps': ddim_steps,
        'sampler_name': sampler_name,
    }

    data = pd.DataFrame(data=[
        # name_column,'模板图片', '模板尺寸', '模板形状', '模板坐标', '商品分类', '推理参数', '评分', '备注', 'date'
        [template_name, template_img, template_size, shape, coordinate, type, json.dumps(inf_params), '', '', datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")],
    ], columns=tm.columns)
    return tm.add(data)

history_dirs = datadir.get_history_dirs()

# # base model
models = sd_models()
models_title = []
commodity_def_model_idx = 0
if models:
    for idx, model in enumerate(models):
        models_title.append(model['title'])
        if model['title'].find(constant.def_model['commodity']) != -1:
            commodity_def_model_idx = idx

# # vae model
vae_models = sd_vae()
vae_models_title = []
commodity_def_vae_idx = 0
if vae_models:
    for idx, model in enumerate(vae_models):
        vae_models_title.append(model['model_name'])
        if model['model_name'].find(constant.def_vae['commodity']) != -1:
            commodity_def_vae_idx = idx




def commodity_tab():
    with gr.Blocks() as G:
        with gr.Blocks() as commodity:
            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem("上传去背后的图片(Remove background result)"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column():
                                        result_rm_img = gr.Image(type='pil', image_mode='RGBA', height=300)
                                with gr.Row():
                                    result_rm_img_but = gr.Button('上传(Upload)', variant='primary')
                            with gr.Column():
                                output_images = gr.Gallery(label='Output', show_label=False, elem_id='rmbg_box',
                                                           columns=1, rows=1, height=300, object_fit="contain")
                    result_rm_img_but.click(fn=upload_rem_img_result, inputs=[result_rm_img], outputs=[output_images])

                    # with gr.TabItem("辅助去背(Remove background)"):
                    #     with gr.Row():
                    #         with gr.Column():
                    #                 gr.Markdown('上传图片(Upload image)')
                    #                 instance_images = gr.Image(type='filepath', height=360)
                    #         with gr.Column():
                    #                 with gr.Column():
                    #                     gr.Markdown('去背结果(Remove background result)')
                    #                     run_button = gr.Button('去背(remove background)')
                    #                 run_button.click(fn=rmbg, inputs=[instance_images], outputs=[output_images])


            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column(min_width=80):
                            template_name = gr.Textbox(label="模板名称", elem_id="template_name", min_width=80)
                        with gr.Column(min_width=80):
                            template_type = gr.Dropdown(label='商品分类', choices=constant.commodity_type, elem_id="commodity_type", value=constant.commodity_type[0], interactive=True, min_width=80)
                        with gr.Column(min_width=80):
                            shape = gr.Dropdown(label='商品形状', choices=constant.commodity_shape, elem_id="commodity_shape", value=constant.commodity_shape[0], min_width=80)
                    with gr.Box():
                        html = """
                        <script>
                        </script>
                            <iframe src="/iframe" width="500" height="700" id='scene_img' style="border:none;"></iframe>
                        """
                        html_element = gr.HTML(html)
                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("参数(Params)"):
                            with gr.Box():
                                with gr.Row():
                                    with gr.Column():
                                        select_model = gr.Dropdown(label='模型(Model)', choices=models_title,
                                                                   elem_id="select_model_list",
                                                                   value=models_title[commodity_def_model_idx] if len(models_title) > 0 else None,
                                                                   interactive=True)
                                    with gr.Column():
                                        mode = gr.Radio(label='作图方式(Mode)', choices=constant.generate_mode, type="value", value=constant.generate_mode[constant.self_innovate_mode], interactive=True)

                            def prompt_change(val):
                                return gr.Textbox.update(value=val)

                            pos_prompt = gr.Textbox(label="提示语(Prompt)", lines=2, elem_id="comm_prompt",
                                                    value=constant.mode_params[constant.self_innovate_mode]['prompt'],
                                                    interactive=True)
                            pos_prompt.change(fn=prompt_change, inputs=[pos_prompt], outputs=[pos_prompt])
                            neg_prompt = gr.Textbox(label="负向提示语(Negative Prompt)", lines=2,
                                                    value=constant.mode_params[constant.self_innovate_mode]['negative_prompt'],
                                                    interactive=True)

                            def g_wh_change(size):
                                print("g_wh_change:", size)
                                return gr.Text.update(value=size)
                            g_width = gr.Text(elem_id="g_width",
                                              value=768, visible=False, interactive=True)
                            g_width.change(g_wh_change, [g_width])
                            g_height = gr.Text(elem_id="g_height",
                                               value=1024, visible=False, interactive=True)
                            g_height.change(g_wh_change, [g_height])
                            with gr.Box():
                                batch_count = gr.Slider(minimum=1, step=1, label='生成数量(Batch count)', value=1,
                                                        elem_id="txt2img_batch_count")
                        with gr.TabItem("高级参数(Advanced Params)"):
                            with gr.Row():
                                select_vae = gr.Dropdown(label='Vae', choices=vae_models_title,
                                                         elem_id="select_vae_list",
                                                         value=vae_models_title[commodity_def_vae_idx] if len(vae_models_title) > 0 else None,
                                                         interactive=True)
                            with gr.Row():
                                with gr.Column(min_width=160):
                                    gr.Markdown('Sampler')
                                    # sampler_name
                                    ddim_steps = gr.Slider(minimum=0, maximum=100, step=1, label='Sampler Steps',
                                                                 value=constant.mode_params[constant.self_innovate_mode]['sampler_step'], elem_id="ddim_steps")
                                    sampler_name = gr.Textbox(label="Sampler name", lines=1, elem_id="sampler_name",
                                                              value='UniPC')

                                with gr.Column(min_width=160):
                                    gr.Markdown('ControlNet')
                                    contr_inp_weight = gr.Slider(minimum=0, maximum=2, step=0.01, label='Inpaint weight',
                                                                 value=constant.mode_params[constant.self_innovate_mode]['inpaint_weight'], elem_id="inpaint_weight")
                                    contr_ipa_weight = gr.Slider(minimum=0, maximum=2, step=0.01, label='IP-Adapter weight',
                                                                 value=constant.mode_params[constant.self_innovate_mode]['ip-adapter_weight'], elem_id="ip_adapter_weight")
                                    contr_lin_weight = gr.Slider(minimum=0, maximum=2, step=0.01, label='Lineart weight',
                                                                 value=constant.mode_params[constant.self_innovate_mode]['lineart_weight'], elem_id="lineart_weight")
                                    contr_scribble_weight = gr.Slider(minimum=0, maximum=2, step=0.01, label='scribble weight',
                                                                 value=constant.mode_params[constant.self_innovate_mode][
                                                                     'scribble_weight'], elem_id="scribble_weight")

                            def mode_change(val):
                                print("mode_change_val:", val)
                                return gr.Slider.update(value=constant.mode_params[val]['inpaint_weight']),\
                                    gr.Slider.update(value=constant.mode_params[val]['ip-adapter_weight']), \
                                    gr.Slider.update(value=constant.mode_params[val]['lineart_weight']),\
                                    gr.Textbox.update(value=constant.mode_params[val]['prompt']), \
                                    gr.Textbox.update(value=constant.mode_params[val]['negative_prompt']), \
                                    gr.Textbox.update(value=constant.mode_params[val]['scribble_weight']), \
                                    gr.Textbox.update(value=constant.mode_params[val]['ddim_step']), \
                                    gr.Textbox.update(value=constant.mode_params[val]['sampler_name']),


                            mode.change(fn=mode_change, inputs=[mode], outputs=[contr_inp_weight,
                                                                                contr_ipa_weight,
                                                                                contr_lin_weight,
                                                                                pos_prompt,
                                                                                neg_prompt,
                                                                                contr_scribble_weight,
                                                                                ddim_steps, sampler_name], queue=False)
                            generate_type = gr.Text(value=1, visible=False, elem_id="generate_type")
                    with gr.Box():
                        with gr.Row():
                            with gr.Column():
                                run_generate = gr.Button('开始制作(Generate)')
                            with gr.Column():
                                template_img = gr.Textbox(label="模板图片", lines=1, elem_id="template_img", visible=False, value='')
                                coordinate = gr.Text(label="坐标", lines=1, elem_id="coordinate", visible=False, value='')
                                coordinate.change(fn=g_wh_change, inputs=[coordinate])
                                template_size = gr.Text(label="模板尺寸", lines=1, elem_id="template_size", visible=False, value='768x1024')
                                template_size.change(fn=g_wh_change, inputs=[template_size])
                                run_save_temp = gr.Button('保存模板(Save)', variant='primary')
            with gr.Box():
                with gr.Row():
                    with gr.Tabs():
                        with gr.TabItem("生成结果(Generate Result)"):
                            with gr.Column():
                                output_generate_images = gr.Gallery(label='Output', show_label=False, columns=4, rows=4, height=500, object_fit="contain")
                        with gr.TabItem('历史记录(History)'):
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column():
                                        total_history = gr.Button('刷新(Refresh)')
                                        history_imgs = gr.Gallery(show_label=True, columns=4, rows=4, height=500)
                                    with gr.Column():
                                        self_innovate_history = gr.Button('刷新(Refresh)')
                                        self_innovate_history_imgs = gr.Gallery(show_label=True, columns=4, rows=4, height=500)
                                        g_type = gr.Textbox(value=3, visible=False)
                        if constant.PT_ENV is not None and constant.PT_ENV != '':
                            with gr.TabItem("api generate result"):
                                    with gr.Box():
                                        def disp_apihistory(id_task):
                                            input = datadir.api_generate_commodity_dir.format(id_task=id_task,
                                                                                              type='input') + '/*.png'
                                            output = datadir.api_generate_commodity_dir.format(id_task=id_task,
                                                                                               type='output') + '/*.png'
                                            return glob.glob(input), glob.glob(output)

                                        with gr.Row():
                                            with gr.Box():
                                                with gr.Column():
                                                    task_id = gr.Text(visible=True, label='Task Id')
                                                with gr.Column():
                                                    api_hist_butt = gr.Button('查看(check)')
                                        with gr.Row():
                                            with gr.Column():
                                                input = gr.Gallery()
                                                pass
                                            with gr.Column():
                                                output = gr.Gallery()
                                                pass
                                        api_hist_butt.click(fn=disp_apihistory, inputs=[task_id], outputs=[input, output])
                            with gr.TabItem("模板实验"):
                                with gr.Box():
                                    # import scripts.templatemanager as templatemanager
                                    stylesdata = gr.Dataframe(
                                        value=tm.get_styles,
                                        col_count=(len(tm.display_columns), 'fixed'),
                                        wrap=True, max_rows=1000, show_label=False, interactive=True,
                                        elem_id="style_editor_grid"
                                    )
                                    stylesdata.input(fn=tm.update_styles, inputs=[stylesdata], outputs=[stylesdata])


            output_message = gr.Markdown()

            run_generate.click(fn=generate,
                               inputs=[mode, select_model, select_vae, pos_prompt, neg_prompt, batch_count,
                                       contr_inp_weight, contr_ipa_weight, contr_lin_weight, generate_type, g_width, g_height, contr_scribble_weight, ddim_steps, sampler_name],
                               outputs=[output_generate_images])
            # template_name, template_img, coordinate, type, pos_prompt, neg_prompt,contr_inp_weight, contr_lin_weight, width, height, contr_scribble_weight, ddim_steps, sampler_name
            run_save_temp.click(fn=save_commdity_tmpe, inputs=[template_name, template_img, template_size, shape, coordinate, template_type, pos_prompt, neg_prompt,contr_inp_weight, contr_lin_weight, g_width, g_height, contr_scribble_weight, ddim_steps, sampler_name], outputs=[stylesdata])
            total_history.click(fn=refresh_history_img, outputs=[history_imgs])
            self_innovate_history.click(fn=refresh_history_img, inputs=[g_type], outputs=[self_innovate_history_imgs])

        return G


def commodity_hand_ui():
    with gr.Blocks() as G:
        with gr.Blocks() as commodity:
            gr.Markdown('实验中...')

        return G

def clothes_upload_file(human, clothes):
    pass


def clothes_ui():
    with gr.Blocks() as G:
        with gr.Blocks():
            with gr.Row():
                with gr.Tabs():

                    with gr.TabItem("上传去背后的图片(Remove background result)"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('上传人物图片(Upload image)')
                                human_image = gr.Image(type='filepath', elem_id="human_image", height=300)
                            with gr.Column():
                                gr.Markdown('上传衣服图片(Upload image)')
                                clothes_image = gr.Image(type='filepath', elem_id="clothes_image", height=300)
                        # with gr.Row():
                            # result_rm_img_but = gr.Button('上传(Upload)')
                            # result_rm_img_but.click(fn=clothes_upload_file, inputs=[human_image, clothes_image])
            with gr.Row():
                with gr.Column():
                    with gr.Box():
                        html = """
                        <script>
                        </script>
                            <iframe src="/iframe_clothes" width="800" height="600" id='scene_img' style="border:none;"></iframe>
                        """
                        html_element = gr.HTML(html)
                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("参数(Params)"):
                            with gr.Box():
                                select_model = gr.Dropdown(label='模型(Model)', choices=models_title,
                                                           elem_id="select_model_list",
                                                           value=models_title[commodity_def_model_idx] if len(models_title) > 0 else None,
                                                           interactive=True)

                            pos_prompt = gr.Textbox(label="提示语(Prompt)", lines=3, elem_id="clothes_prompt",
                                                    value=prompt,
                                                    interactive=True)
                            neg_prompt = gr.Textbox(label="负向提示语(Negative Prompt)", lines=3,
                                                    value=negative_prompt_clothes,
                                                    interactive=True)
                            with gr.Box():
                                batch_count = gr.Slider(minimum=1, step=1, label='生成数量(Batch count)', value=1,
                                                        elem_id="txt2img_batch_count")
                        with gr.TabItem("高级参数(Advanced Params)"):
                            with gr.Column():
                                select_vae = gr.Dropdown(label='Vae', choices=vae_models_title,
                                                         elem_id="select_vae_list",
                                                         value=vae_models_title[commodity_def_vae_idx] if len(vae_models_title) > 0 else None,
                                                         interactive=True)
                            gr.Markdown('ControlNet')
                            contr_inp_weight = gr.Slider(minimum=0, maximum=2, step=0.01, label='Inpaint weight',
                                                         value=0.5, elem_id="inpaint_weight")
                            contr_ipa_weight = gr.Slider(minimum=0, maximum=2, step=0.01, label='IP-Adapter weight',
                                                         value=0.55, elem_id="ip_adapter_weight")
                            contr_lin_weight = gr.Slider(minimum=0, maximum=2, step=0.01, label='Lineart weight',
                                                         value=0.7, elem_id="lineart_weight")

                    run_generate = gr.Button('开始制作(Generate)')
            with gr.Box():
                with gr.Row():
                    with gr.Tabs():
                        with gr.TabItem("生成结果(Generate Result)"):
                            with gr.Column():
                                output_generate_images = gr.Gallery(label='Output', show_label=False, columns=4, rows=4, height=500, object_fit="contain")
                        with gr.TabItem('历史记录(History)'):
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column():
                                        total_history = gr.Button('刷新(Refresh)')
                                history_imgs = gr.Gallery(show_label=True, columns=4, rows=4, height=500)

            generate_type = gr.Text(value=2, visible=False)

            run_generate.click(fn=generate_image,
                               inputs=[select_model, select_vae, pos_prompt, neg_prompt, batch_count,
                                       contr_inp_weight, contr_ipa_weight, contr_lin_weight, generate_type],
                               outputs=[output_generate_images])
            total_history.click(fn=refresh_history_img, inputs=[generate_type], outputs=[history_imgs])
    return G


with gr.Blocks() as G:
    with gr.Tabs():
        with gr.TabItem('商品图'):
            commodity_tab()
        with gr.TabItem("手持商品图"):
            commodity_hand_ui()
        with gr.TabItem("换装"):
            clothes_ui()

if __name__ == '__main__':
    #  max_size=2, api_open=False, status_update_rate='auto'
    app, local_url, share_url = G.queue(concurrency_count=64).launch(server_name=cmd_opts.ip, server_port=cmd_opts.port, show_error=True, share=cmd_opts.share, prevent_thread_lock=True)
    # enable_queue=True,
    app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']
    Api.Api(app)
    import asyncio
    asyncio.run(Api.call_queue_task())
    while 1:
        time.sleep(1)
