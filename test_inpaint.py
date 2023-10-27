from scripts.inpaint import run_inpaint
from utils.utils import project_dir


# 原生sd和diffusers模型之间转换
# https://zhuanlan.zhihu.com/p/645757706
if __name__ == "__main__":
    # 输入图像
    input_image = f'{project_dir}/test/input/inpaint/inpaint_image.png'
    # 输入mask
    mask_image = f'{project_dir}/test/input/inpaint/inpaint_image_mask.png'
    prompt = 'a man in a suit and tie posing for a picture with his hands in his pockets and his shirt tucked into his waist,In front of the villa,(high_contrast:1.2), vivid photo effect, RAW photo,realistic,dramatic lighting,ultra high res,best quality,high quality'
    n_prompt = 'nsfw,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4),(deformed, distorted, disfigured:1.3),poorly drawn,bad anatomy,wrong anatomy,extra limb,missing limb,floating limbs,disconnected limbs,mutation,mutated,ugly,disgusting,amputation'
    ddim_steps = 25
    cfg_scale = 7.5
    seed = -1
    inp_model_id = 'Uminosachi/realisticVisionV51_v51VAE-inpainting'
    composite_chk = True
    controlnets = [
        {
            # 'model_path': 'lllyasviel/sd-controlnet-canny',
            # Please make sure to pass `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize those weights or else make sure your checkpoint file is correct.
            'low_cpu_mem_usage': False,
            'device_map': None,
            'model_path': '/data/aigc/stable-diffusion-webui/extensions/sd-webui-controlnet/models/ip-adapter-plus/',
            'scale': 0.6,
            'image': f'{project_dir}/test/input/inpaint/inpaint_image.png'
        },
        {
            'model_path': '/data/aigc/stable-diffusion-webui/extensions/sd-webui-controlnet/models/lineart-fp16/',
            'scale': 0.6,
            'image': f'{project_dir}/test/input/inpaint/lineart.png'
        }
    ]
    a,b = run_inpaint(input_image, mask_image, prompt, n_prompt, ddim_steps, cfg_scale, seed, inp_model_id, composite_chk, controlnets=controlnets)
    print(a,b)

