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
        with open(image, 'rb') as f:
            content = f.read()
            with open(commodity_image_path, 'wb') as ff:
                ff.write(content)
                ff.close()
            with open(commodity_rembg_image_path, 'wb') as ffs:
                ffs.write(content)
                ffs.close()
        return [commodity_rembg_image_path]
    else:
        with open(image, 'rb') as f:
            with open(commodity_image_path, 'wb') as ff:
                ff.write(f.read())
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
    if type==1:
        dirs = datadir.generate_glob_img
    else:
        dirs = datadir.clothes_generate_glob_img
    return glob.glob(dirs)


def upload_rem_img_result(img):
    # return rmbg(img, True)
    path = rmbg(img, True)
    # print(f"path:{path}")
    # return  path
    # output_images.update(value=path)
    return gr.Gallery.update(value=path, visible=True)
    # return gr.Gallery.update(value=path[0], visible=True)


history_dirs = datadir.get_history_dirs()

# # base model
models = sd_models()
models_title = []
commodity_def_model_idx = 0
print(f"models:{models}")
if models:
    for idx, model in enumerate(models):
        models_title.append(model['title'])
        if model['title'].find(constant.def_model['commodity']) != -1:
            commodity_def_model_idx = idx

# # vae model
vae_models = sd_vae()
vae_models_title = []
commodity_def_vae_idx = 0
print(f"vae_models:{vae_models}")
if vae_models:
    for idx, model in enumerate(vae_models):
        vae_models_title.append(model['model_name'])
        if model['model_name'].find(constant.def_vae['commodity']) != -1:
            commodity_def_vae_idx = idx

print("vae_models_title:", vae_models_title)
def commodity_tab():
    with gr.Blocks() as G:
        with gr.Blocks() as commodity:
            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem("辅助去背(Remove background)"):
                        with gr.Row():
                            with gr.Column():
                                    gr.Markdown('上传图片(Upload image)')
                                    instance_images = gr.Image(type='filepath').style(height=360)
                            with gr.Column():
                                    with gr.Column():
                                        gr.Markdown('去背结果(Remove background result)')
                                        run_button = gr.Button('去背(remove background)')
                                        output_images = gr.Gallery(label='Output', show_label=False, elem_id='rmbg_box').style(columns=1, rows=1, height=300, object_fit="contain")
                                    run_button.click(fn=rmbg, inputs=[instance_images], outputs=[output_images])

                    with gr.TabItem("上传去背后的图片(Remove background result)"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('上传图片(Upload image)')
                                result_rm_img = gr.Image(type='filepath').style(height=300)
                        with gr.Row():
                            result_rm_img_but = gr.Button('上传(Upload)')
                            result_rm_img_but.click(fn=upload_rem_img_result, inputs=[result_rm_img], outputs=[output_images])
            with gr.Row():
                with gr.Column():
                    with gr.Box():
                        html = """
                        <script>
                        </script>
                            <iframe src="/iframe" width="500" height="600" id='scene_img' style="border:none;"></iframe>
                        """
                        html_element = gr.HTML(html)
                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("参数(Params)"):
                            with gr.Box():
                                select_model = gr.Dropdown(label='模型(Model)', choices=models_title,
                                                           elem_id="select_model_list",
                                                           value=models_title[commodity_def_model_idx],
                                                           interactive=True).style(width=50)
                            def prompt_change(val):
                                print(2222)
                                return gr.Textbox.update(value=val)

                            pos_prompt = gr.Textbox(label="提示语(Prompt)", lines=3, elem_id="comm_prompt",
                                                    value=prompt,
                                                    interactive=True)
                            pos_prompt.change(prompt_change, [pos_prompt])
                            neg_prompt = gr.Textbox(label="负向提示语(Negative Prompt)", lines=3,
                                                    value=negative_prompt,
                                                    interactive=True)

                            def g_wh_change(size):
                                return gr.Text.update(value=size)
                            g_width = gr.Text(elem_id="g_width",
                                              value=768, visible=True, interactive=True)
                            g_width.change(g_wh_change, [g_width])
                            g_height = gr.Text(elem_id="g_height",
                                               value=1024, visible=True, interactive=True)
                            g_height.change(g_wh_change, [g_height])
                            with gr.Box():
                                batch_count = gr.Slider(minimum=1, step=1, label='生成数量(Batch count)', value=1,
                                                        elem_id="txt2img_batch_count")
                        with gr.TabItem("高级参数(Advanced Params)"):
                            with gr.Column():
                                select_vae = gr.Dropdown(label='Vae', choices=vae_models_title,
                                                         elem_id="select_vae_list",
                                                         value=vae_models_title[commodity_def_vae_idx],
                                                         interactive=True).style(width=50)
                            gr.Markdown('ControlNet')
                            contr_inp_weight = gr.Slider(minimum=0, maximum=2, step=0.01, label='Inpaint weight',
                                                         value=0.5, elem_id="inpaint_weight")
                            contr_ipa_weight = gr.Slider(minimum=0, maximum=2, step=0.01, label='IP-Adapter weight',
                                                         value=0.55, elem_id="ip_adapter_weight")
                            contr_lin_weight = gr.Slider(minimum=0, maximum=2, step=0.01, label='Lineart weight',
                                                         value=0.7, elem_id="lineart_weight")
                            generate_type = gr.Text(value=1, visible=False, elem_id="generate_type")
                    run_generate = gr.Button('开始制作(Generate)')
            with gr.Box():
                with gr.Row():
                    with gr.Tabs():
                        with gr.TabItem("生成结果(Generate Result)"):
                            with gr.Column():
                                output_generate_images = gr.Gallery(label='Output', show_label=False).style(
                                    columns=4, rows=4,
                                    height=500,
                                    object_fit="contain")
                        with gr.TabItem('历史记录(History)'):
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column():
                                        total_history = gr.Button('刷新(Refresh)').style(height=10)
                                history_imgs = gr.Gallery(show_label=True).style(columns=4, rows=4, height=500)

            output_message = gr.Markdown()

            run_generate.click(fn=generate_image,
                               inputs=[select_model, select_vae, pos_prompt, neg_prompt, batch_count,
                                       contr_inp_weight, contr_ipa_weight, contr_lin_weight, generate_type, g_width, g_height],
                               outputs=[output_generate_images])
            total_history.click(fn=refresh_history_img, outputs=[history_imgs])

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
                                human_image = gr.Image(type='filepath', elem_id="human_image").style(height=300)
                            with gr.Column():
                                gr.Markdown('上传衣服图片(Upload image)')
                                clothes_image = gr.Image(type='filepath', elem_id="clothes_image").style(height=300)
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
                                                           value=models_title[commodity_def_model_idx],
                                                           interactive=True).style(width=50)

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
                                                         value=vae_models_title[commodity_def_vae_idx],
                                                         interactive=True).style(width=50)
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
                                output_generate_images = gr.Gallery(label='Output', show_label=False).style(
                                    columns=4, rows=4,
                                    height=500,
                                    object_fit="contain")
                        with gr.TabItem('历史记录(History)'):
                            with gr.Column():
                                with gr.Row():
                                    with gr.Column():
                                        total_history = gr.Button('刷新(Refresh)').style(height=10)
                                history_imgs = gr.Gallery(show_label=True).style(columns=4, rows=4, height=500)

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
    app, local_url, share_url = G.queue(64).launch(server_name=cmd_opts.ip, server_port=cmd_opts.port, show_error=True,
                                                   prevent_thread_lock=True)
    app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']
    Api.Api(app)
    while 1:
        time.sleep(0.5)
