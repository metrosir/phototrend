from controlnet_aux import LineartDetector
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

def get_lineart_image(image, image_resolution, model_path='lllyasviel/Annotators'):
    lineart = LineartDetector.from_pretrained(model_path, filename='')
    return lineart(image,image_resolution=image_resolution, output_type="pil")


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    # todo
    input_image = np.array(input_image, dtype=np.uint8)
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad


def lineart_standard(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    x = img.astype(np.float32)
    g = cv2.GaussianBlur(x, (0, 0), 6.0)
    intensity = np.min(g - x, axis=2).clip(0, 255)
    intensity /= max(16, np.median(intensity[intensity > 8]))
    intensity *= 127
    result = intensity.clip(0, 255).astype(np.uint8)
    return remove_pad(result), True


model_lineart = None


def lineart(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_lineart
    if model_lineart is None:
        from .lineart import LineartDetector
        model_lineart = LineartDetector(LineartDetector.model_default)

    # applied auto inversion
    result = 255 - model_lineart(img)
    return remove_pad(result), True


def lineart_image(input_image, width):
    if type(input_image) is str:
        input_image = Image.open(BytesIO(open(input_image, 'rb').read())).convert("RGB")
    # width, height = img.size
    input_image, x = lineart(input_image, res=width)
    input_image = input_image.astype(np.uint8)
    input_image = Image.fromarray(input_image)
    # image.save(f'{project_dir}/test/input/inpaint/linearts/2_test1.png')
    return input_image

def scribble_xdog(img, res=512, thr_a=32, **kwargs):
    if type(img) is str:
        img = Image.open(BytesIO(open(img, 'rb').read())).convert("RGB")
    img, remove_pad = resize_image_with_pad(img, res)
    g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.5)
    g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 5.0)
    dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
    result = np.zeros_like(img, dtype=np.uint8)
    result[2 * (255 - dog) > thr_a] = 255
    input_image = remove_pad(result)
    input_image = input_image.astype(np.uint8)
    input_image = Image.fromarray(input_image)
    return input_image


# def pidinet(img, res=512, **kwargs):
#     img, remove_pad = resize_image_with_pad(img, res)
#     global model_pidinet
#     if model_pidinet is None:
#         from annotator.pidinet import apply_pidinet
#         model_pidinet = apply_pidinet
#     result = model_pidinet(img)
#     return remove_pad(result), True
#
# def scribble_pidinet(img, res=512, **kwargs):
#     result, _ = pidinet(img, res)
#     import cv2
#     from annotator.util import nms
#     result = nms(result, 127, 3.0)
#     result = cv2.GaussianBlur(result, (0, 0), 3.0)
#     result[result > 4] = 255
#     result[result < 255] = 0
#     return result, True