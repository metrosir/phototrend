from scripts.inpaint import run_inpaint, Inpainting
from utils.utils import project_dir

def inpaint3():
    input_image = f'{project_dir}/test/input/inpaint/images/2.png'
    input_lineart_image = f'{project_dir}/test/input/inpaint/linearts/2.png'
    mask_image = f'{project_dir}/test/input/inpaint/masks/2.png'
    ip = Inpainting(
                # base_model="Uminosachi/realisticVisionV51_v51VAE-inpainting",
                base_model="/data/aigc/stable-diffusion-webui/models/Stable-diffusion/civitai/residentchiefnz/diffusers",
                controlnet=[
                    {
                        # 'model_path': 'lllyasviel/sd-controlnet-canny',
                        # Please make sure to pass `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize those weights or else make sure your checkpoint file is correct.
                        'low_cpu_mem_usage': False,
                        'device_map': None,
                        'model_path': '/data/aigc/stable-diffusion-webui/extensions/sd-webui-controlnet/models/ip-adapter-plus/',
                        'scale': 0.55,
                        'image': input_image
                    },
                    {
                        'model_path': '/data/aigc/stable-diffusion-webui/extensions/sd-webui-controlnet/models/lineart-fp16/',
                        'scale': 0.7,
                        'image': input_lineart_image
                    }
                ]
               )
    ip.set_textual_inversion(
        '/data/aigc/stable-diffusion-webui/embeddings/negative/realisticvision-negative-embedding.pt',
        'realisticvision-negative-embedding',
        'string_to_param'
    )
    ip.run_inpaint(
        input_image=input_image,
        mask_image=mask_image,
        prompt="a bottle of oligosog beauty oil sitting on a table next to flowers and a vase with white flowers,(high_contrast), RAW photo,realistic,dramatic lighting,ultra high res,best quality,high quality,",
        n_prompt='realisticvision-negative-embedding',
        ddim_steps=40,
        cfg_scale=7.5,
        seed=-1,
        composite_chk=True,
        # sampler_name="Euler a",
        sampler_name="Euler a",
        iteration_count=1
        )





# 原生sd和diffusers模型之间转换
# python convert_original_stable_diffusion_to_diffusers.py  --checkpoint_path /data/aigc/stable-diffusion-webui/models/Stable-diffusion/civitai/residentchiefnz/icbinpRelapseRC.zgS8.safetensors --dump_path /data/aigc/stable-diffusion-webui/models/Stable-diffusion/civitai/residentchiefnz/diffusers/  --image_size 512 --prediction_type epsilon --from_safetensors
# https://zhuanlan.zhihu.com/p/645757706
def inpaint():
    # 输入图像
    input_image = f'{project_dir}/test/input/inpaint/images/1.png'
    input_lineart_image = f'{project_dir}/test/input/inpaint/linearts/1.png'

    # 输入mask
    mask_image = f'{project_dir}/test/input/inpaint/masks/1.png'

    # prompt = 'a man in a suit and tie posing for a picture with his hands in his pockets and his shirt tucked into his waist,In front of the villa,(high_contrast:1.2), vivid photo effect, RAW photo,realistic,dramatic lighting,ultra high res,best quality,high quality'
    prompt = "a hand holding a bottle of skin care product in it's palm, hand101,folded fingers:1.6,fingers folded,folded fingers, (high_contrast), RAW photo,realistic,dramatic lighting,ultra high res,best quality,high quality"
    n_prompt = 'realisticvision-negative-embedding'
    ddim_steps = 16
    cfg_scale = 7.5
    seed = -1
    inp_model_id = 'Uminosachi/realisticVisionV51_v51VAE-inpainting'
    composite_chk = True
    controlnets = [
        # {
        #     # 'model_path': 'lllyasviel/sd-controlnet-canny',
        #     # Please make sure to pass `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize those weights or else make sure your checkpoint file is correct.
        #     'low_cpu_mem_usage': False,
        #     'device_map': None,
        #     'model_path': '/data/aigc/stable-diffusion-webui/extensions/sd-webui-controlnet/models/ip-adapter-plus/',
        #     'scale': 0.3,
        #     'image': input_image
        # },
        {
            'model_path': '/data/aigc/stable-diffusion-webui/extensions/sd-webui-controlnet/models/lineart-fp16/',
            'scale': 1,
            'image': input_lineart_image
        }
    ]
    a,b = run_inpaint(input_image, mask_image, prompt, n_prompt, ddim_steps, cfg_scale, seed, inp_model_id, composite_chk, controlnets=controlnets)
    print(a,b)



def inpaint2():
    from diffusers import ControlNetModel, DDIMScheduler, DiffusionPipeline, StableDiffusionControlNetInpaintPipeline
    import torch
    import cv2
    import numpy as np
    import urllib
    import os
    from PIL import Image
    from utils.datadir import generate_inpaint_image_dir

    device = "cuda"

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        # "runwayml/stable-diffusion-inpainting",
        "Uminosachi/realisticVisionV51_v51VAE-inpainting",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    img_url = "https://i.ibb.co/0ZK7yL0/img.png"
    img_red_bg_url = "https://i.ibb.co/zHHZTkZ/img-red-bg.png"
    canny_url = "https://i.ibb.co/rp9FYCX/canny.png"
    mask_url = "https://i.ibb.co/FK4DNNK/mask.png"

    def decode_image(image_url):
        print("Decode image: " + image_url)
        req = urllib.request.urlopen(image_url)
        arr = np.array(bytearray(req.read()), dtype=np.uint8)
        imageBGR = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGRA2RGB)

        return Image.fromarray(imageRGB)

    img = decode_image(img_url)
    img_red_bg = decode_image(img_red_bg_url)
    canny = decode_image(canny_url)
    mask = decode_image(mask_url)

    with torch.inference_mode():
        output = pipe(
            prompt="product in a forest",
            negative_prompt="",
            image=img_red_bg,
            mask_image=mask,
            control_image=canny,
            height=1024,
            width=768,
            num_images_per_prompt=1,
            num_inference_steps=20,
            guidance_scale=7.5,
            strength=1
        )

        output_list = []
        for output_image in output.images:
            img_idx = len(os.listdir(generate_inpaint_image_dir))

            # save_name = generate_inpaint_image_dir + f"{img_idx}.png"
            save_name = os.path.join(generate_inpaint_image_dir, f"{img_idx}.png")
            # save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
            output_image.save(save_name)

            output_list.append(output_image)
            print("Saved image: " + save_name)


if __name__ == "__main__":
    inpaint3()