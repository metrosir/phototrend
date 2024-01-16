from diffusers import (StableDiffusionControlNetInpaintPipeline, ControlNetModel)


def load(pipe, model_id, token: str, weight_name: str, subfolder: str = None):

    pipe = pipe.load_textual_inversion(
        pretrained_model_name_or_path=model_id,
        token=token,
        weight_name=weight_name,
        subfolder=subfolder,
        # torch_dtype=torch_dtype,
    )
    return pipe

def debug_load(file_path: str):
    import torch

    ckpt = torch.load(file_path)
    return ckpt


if __name__ == '__main__':
    # debug
    import torch
    model_id = '/data/aigc/stable-diffusion-webui/embeddings/negative/realisticvision-negative-embedding.pt' # string_to_param
    # try:
    #     controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    #     pip = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    #         'Uminosachi/realisticVisionV51_v51VAE-inpainting', torch_dtype=torch.float16, local_files_only=True, controlnet=controlnet)
    # except Exception as e:
    #     print("Error:", str(e))
    #     sys.exit(1)

    result = debug_load(model_id)
    print(result)
