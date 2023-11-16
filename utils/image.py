
from PIL import Image, PngImagePlugin
import base64
from rembg import remove
from utils.pt_logging import ia_logging
from torchvision import transforms
import numpy as np


# 制作mask
def convert_png_to_mask(file_path, save_path):
    # 打开图像并转换为RGBA模式
    img = Image.open(file_path)
    rgba = img.convert("RGBA")

    # 获取图像的每个像素的RGBA值
    datas = rgba.getdata()
    new_data = []
    for item in datas:
        # 改变所有透明（也即A<255）的像素为黑色,
        # 将所有不完全不透明的像素都视为透明像素。如果希望只有完全透明的像素被视为透明，可以将if item[3] < 255:改为if item[3] == 0:
        # if item[3] < 255:
        if item[3] == 0:
            new_data.append((0, 0, 0, 255))
        else:
            new_data.append((255, 255, 255, 255))

    # 更新图像数据
    rgba.putdata(new_data)

    # 保存图像
    rgba.save(save_path, "PNG")


def image_to_base64(img_path):
    res = ''
    with open(img_path, "rb") as image_file:
        res = base64.b64encode(image_file.read()).decode('utf-8')
    if res != '':
        res = f'data:image/png;base64,{res}'
        return res
    return False


def remove_bg(inputim, outputim, mask=False, alpha_matting=True):
    with open(inputim, 'rb') as f:
        with open(outputim, 'wb') as ff:
            input=f.read()
            output=remove(input, only_mask=mask, alpha_matting=alpha_matting)
            ff.write(output)


def is_webp_image(image_path):
    try:
        image = Image.open(image_path)
        return image.format == "WEBP"
    except IOError:
        return False


def open_image_to_pil(image_path, convert='RGB'):
    image = Image.open(image_path).convert(convert)
    return image


def save_webp_image_with_transparency(image_path, save_path):
    if is_webp_image(image_path):
        image = Image.open(image_path)
        new_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        new_image.paste(image, (0, 0), image)
        new_image.save(save_path, format="WEBP", transparency=0)
        return True
    return False

def read_image_to_np(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    return image


def auto_resize_to_pil(input_image, mask_image):
    init_image = Image.fromarray(input_image).convert("RGB")
    mask_image = Image.fromarray(mask_image).convert("RGB")
    assert init_image.size == mask_image.size, "The sizes of the image and mask do not match"
    width, height = init_image.size

    #
    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    if new_width < width or new_height < height:
        if (new_width / width) < (new_height / height):
            scale = new_height / height
        else:
            scale = new_width / width
        resize_height = int(height*scale+0.5)
        resize_width = int(width*scale+0.5)
        if height != resize_height or width != resize_width:
            ia_logging.info(f"resize: ({height}, {width}) -> ({resize_height}, {resize_width})")
            init_image = transforms.functional.resize(init_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
            mask_image = transforms.functional.resize(mask_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
        if resize_height != new_height or resize_width != new_width:
            ia_logging.info(f"center_crop: ({resize_height}, {resize_width}) -> ({new_height}, {new_width})")
            init_image = transforms.functional.center_crop(init_image, (new_height, new_width))
            mask_image = transforms.functional.center_crop(mask_image, (new_height, new_width))

    return init_image, mask_image


def mask_invert(mask_img, mask_img_invert):
    # pip install Pillow
    from PIL import Image
    # pip install numpy
    import numpy as np

    # 打开图片
    img = Image.open(mask_img).convert('L')
    # 将图片转换为numpy数组
    img_np = np.array(img)
    # 对图片进行反转
    img_np = 255 - img_np
    # 将反转后的numpy数组转回图片
    img_inverted = Image.fromarray(img_np)
    # 保存反转后的图片
    img_inverted.save(mask_img_invert)


from io import BytesIO
import io


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]

    image = Image.open(BytesIO(base64.b64decode(encoding))).convert("RGB")

    # image = np.array(image)
    return image


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        use_metadata = False
        metadata = PngImagePlugin.PngInfo()
        for key, value in image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True
        image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=100)

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)


def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return encode_pil_to_base64(pil)


def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return encode_pil_to_base64(image)
    elif type(image) is np.ndarray:
        return encode_np_to_base64(image)
    else:
        return ""