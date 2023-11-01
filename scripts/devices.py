import sys
import contextlib
import torch
import scripts.errors as errors


def extract_device_id(args, name):
    for x in range(len(args)):
        if name in args[x]:
            return args[x + 1]

    return None


def enable_tf32():
    if torch.cuda.is_available():
        if any([torch.cuda.get_device_capability(devid) == (7, 5) for devid in range(0, torch.cuda.device_count())]):
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")
cpu = torch.device("cpu")
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16
dtype_unet = torch.float16
unet_needs_upcast = False


def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


def randn(seed, shape):
    torch.manual_seed(seed)
    if device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    if device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)


def autocast(disable=False):

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32:
        return contextlib.nullcontext()

    return torch.autocast("cuda")


def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass