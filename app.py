import gradio as gr
import pathlib
import uuid
import os
import glob
import utils.datadir as datadir
from utils.req import generate_image, prompt, negative_prompt
from utils.image import remove_bg, convert_png_to_mask
import api.api as Api
import time


def rmbg(image) -> str:
    if not os.path.exists(path=datadir.commodity_image_dir):
        pathlib.Path(datadir.commodity_image_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(datadir.commodity_rembg_image_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(datadir.commodity_rembg_mask_image_dir).mkdir(parents=True, exist_ok=True)

    commodity_image_path = f'{datadir.commodity_image_dir}/{datadir.get_file_idx(True)}.png'
    commodity_rembg_image_path = f'{datadir.commodity_rembg_image_dir}/{datadir.get_file_idx(True)}.png'
    commodity_rembg_mask_image_path = f'{datadir.commodity_rembg_mask_image_dir}/{datadir.get_file_idx(True)}.png'
    with open(image, 'rb') as f:
        with open(commodity_image_path, 'wb') as ff:
            ff.write(f.read())
    remove_bg(commodity_image_path, commodity_rembg_image_path)
    remove_bg(commodity_rembg_image_path, commodity_rembg_mask_image_path, True)
    # convert_png_to_mask(commodity_rembg_image_path, commodity_rembg_mask_image_path)
    # ! rembg i -a $commodity_image_path $commodity_rembg_image_path
    # ! rembg i -a -om $commodity_image_path $commodity_rembg_mask_image_path
    yield ["去背完成", [commodity_rembg_image_path, commodity_rembg_mask_image_path]]


def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths


def refresh_history_img():
    return glob.glob(datadir.generate_glob_img)


with gr.Blocks() as G:
    gr.Markdown("商品图")
    with gr.Tabs():
        with gr.TabItem('\N{rocket} 商品图'):
            with gr.Blocks() as commodity:
                with gr.Row():
                    with gr.Column():
                        instance_images = gr.Image(type='filepath').style(height=300)
                    with gr.Column():
                        gr.Textbox(label="任务ID", value=datadir.uuid, interactive=False)
                        run_button = gr.Button('去背(remove background)')
                        infer_progress = gr.Textbox(label="去背进度(Progress)", value="当前无生成任务(No task)",
                                                    interactive=False)
                with gr.Row():
                    with gr.Column():
                        with gr.Box():
                            gr.Markdown('去背结果(Result)')
                            output_images = gr.Gallery(label='Output', show_label=False, elem_id='rmbg_box').style(
                                columns=3, rows=1,
                                height=570, object_fit="contain")
                    with gr.Column():
                        with gr.Box():
                            html = """
                            <script>

                            </script>
                                <iframe src="/iframe" width="500" height="600" id='scene_img' style="border:none;"></iframe>
                            """
                            html_element = gr.HTML(html)
                    with gr.Column():
                        pos_prompt = gr.Textbox(label="提示语(Prompt)", lines=3, elem_id="comm_prompt",
                                                value=prompt,
                                                interactive=True)
                        neg_prompt = gr.Textbox(label="负向提示语(Negative Prompt)", lines=3,
                                                value=negative_prompt,
                                                interactive=True)
                        with gr.Column(elem_id="txt2img_column_batch"):
                            batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1,
                                                    elem_id="txt2img_batch_count")
                        run_generate = gr.Button('开始制作(Generate)')
                        with gr.Box():
                            generate_progress = gr.Textbox(label="生成进度(Progress)", value="当前无生成任务(No task)",
                                                           interactive=False)
                with gr.Box():
                    gr.Markdown('生成结果(Generate Result)')
                    output_generate_images = gr.Gallery(label='Output', show_label=False).style(columns=4, rows=4,
                                                                                                height=500,
                                                                                                object_fit="contain")
                with gr.Box():
                    with gr.Accordion(label='历史记录(History)', open=False):
                        refresh_history = gr.Button('刷新(Refresh)')
                        history_generate_images = gr.Gallery(fn=refresh_history_img).style(grid=8, height=600)
                output_message = gr.Markdown()

                run_button.click(fn=rmbg, inputs=[instance_images], outputs=[infer_progress, output_images])

                run_generate.click(fn=generate_image, inputs=[pos_prompt, neg_prompt, batch_count],
                                   outputs=[generate_progress, output_generate_images])
                refresh_history.click(fn=refresh_history_img, inputs=[], outputs=[history_generate_images])



if __name__ == '__main__':
    app, local_url, share_url = G.queue(64).launch(server_name="10.61.158.18", server_port=8080, show_error=True, prevent_thread_lock=True)
    app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']
    Api.Api(app)
    while 1:
        time.sleep(0.5)