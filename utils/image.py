
from PIL import Image
import base64
from rembg import remove


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
