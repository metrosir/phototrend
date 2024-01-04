
import os
import threading
import utils.pt_logging as pt_logging
from PIL import Image
import numpy as np
import cv2
import asyncio

from utils.image import image_to_base64
from utils.constant import PT_ENV


def add_background_color(image_path, bg_color):
    img = Image.open(image_path)
    bg = Image.new('RGB', img.size, bg_color)
    bg.paste(img, (0, 0), img)
    bg.save(image_path)
    return image_path

class Imageshadowss:
    def __init__(self, x_offset, y_offset, blur, opacity, toggle=0, color="#000000", bg_color="#ffffff"):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.blur = blur
        self.color = color
        self.opacity = opacity
        self.toggle = toggle
        self.bg_color = bg_color

    async def __call__(self, img_input_path, img_output_path, ret_base64=False):
        try:
            # lock = threading.Lock()
            # lock.acquire()
            # infile outfile x y blur color opacity toggle
            cmd=[
                "flatpak", "run", "org.gimp.GIMP//stable", "-i", "-b", f"(add-shadow \"{img_input_path}\" \"{img_output_path}\" {self.x_offset} {self.y_offset} {self.blur} \"{self.color}\" {self.opacity} {self.toggle} \"{self.bg_color}\")", "-b", "(gimp-quit 0)"
                # "flatpak run org.gimp.GIMP//stable -i -b '(add-shadow \"{img_input_path}\" \"{img_output_path}\" {self.x_offset} {self.y_offset} {self.blur} \"{self.color}\" {self.opacity} {self.toggle} \"{self.bg_color}\")' -b '(gimp-quit 0)'"
            ]
            pt_logging.ia_logging.info(cmd)
            # os.system(cmd)
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.communicate()
            # lock.release()
            add_background_color(img_output_path, self.bg_color)
            if ret_base64:
                return image_to_base64(img_output_path, False)
        except Exception as e:
            pt_logging.log_echo(
                title="add-shadow error",
                msg={"img_input_path": img_input_path, "img_output_path": img_output_path},
                exception=e,
                level="error",
            )
            raise e
        finally:
            pass


class ImagePerspectiveShadow:

    # 角度、水平的相对距离、阴影的相对长度、模糊半径、颜色、不透明度、插值、允许改变大小
    def __init__(self, v_angle, x_distance, shadow_length, blur, opacity, bg_color, p_gradient_strength,  toggle=0, allow_update_size=0, color="#000000"):
        self.v_angle = v_angle

        self.x_distance = x_distance
        self.shadow_length = shadow_length
        self.blur = blur
        self.color = color
        self.opacity = opacity
        self.toggle = toggle
        # self.gradient = 'Flare Rays Radial 1'
        if PT_ENV == 'ptdev' or PT_ENV == 'pt_dev':
            self.gradient = '前景到透明'
        else:
            self.gradient = 'FG to Transparent'
        self.gradient_strength = p_gradient_strength
        self.bg_color = bg_color
        if allow_update_size is None or allow_update_size == False:
            allow_update_size = 0
        else:
            allow_update_size = 1
        self.allow_update_size = allow_update_size

    async def __call__(self, img_input_path, img_output_path, ret_base64=False):
        try:
            orientation = self.check_image_orientation(img_input_path)
            if orientation is None:
                raise Exception("image mode is not RGBA")
            if orientation == 'w':
                self.x_distance = 5
            # lock = threading.Lock()
            # lock.acquire()
            # v_angle = self.get_angle(img_input_path)

            # infile outfile x y blur color opacity toggle
            cmd=[
                # f"unset LD_PRELOAD;flatpak run org.gimp.GIMP//stable -i -b '(add-perspective-shadow \"{img_input_path}\" \"{img_output_path}\" {self.v_angle} {self.x_distance} {self.shadow_length} {self.blur} \"{self.color}\" {self.opacity} {self.toggle}  {self.allow_update_size} \"{self.gradient}\" \"{self.bg_color}\" {self.gradient_strength})' -b '(gimp-quit 0)'",
                "flatpak", "run", "org.gimp.GIMP//stable", "-i", "-b", f"(add-perspective-shadow \"{img_input_path}\" \"{img_output_path}\" {self.v_angle} {self.x_distance} {self.shadow_length} {self.blur} \"{self.color}\" {self.opacity} {self.toggle}  {self.allow_update_size} \"{self.gradient}\" \"{self.bg_color}\" {self.gradient_strength})", "-b", "(gimp-quit 0)"
            ]
            pt_logging.ia_logging.info(cmd)
            # os.system(cmd)
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.communicate()

            # lock.release()
            add_background_color(img_output_path, self.bg_color)
            if ret_base64:
                return image_to_base64(img_output_path, False)
        except Exception as e:
            pt_logging.log_echo(
                title="add-shadow error",
                msg={"img_input_path": img_input_path, "img_output_path": img_output_path},
                exception=e,
                level="error",
            )
            raise e
        finally:
            pass


    # 获取旋转角度
    def get_angle(self, img):
        img = cv2.imread(img)
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 应用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 初始化最大面积和对应的角度
        max_area = 0
        max_angle = 0
        for cnt in contours:
            # 计算轮廓面积
            area = cv2.contourArea(cnt)
            # 如果当前轮廓面积大于最大面积，则更新最大面积和对应的角度
            if area > max_area:
                max_area = area
                # 计算最小面积矩形
                rect = cv2.minAreaRect(cnt)
                # 获取旋转角度
                angle = rect[-1]
                if angle < -45:
                    angle = 90 + angle

                max_angle = angle
        return max_angle

    def get_image_isoverlook(self, img):
        '''
        判断是否为俯视拍摄的物体
        对物体的轮廓进行分析，通过轮廓的形状、比例等特征来判断物体的姿态。
        '''
        img = cv2.imread(img)
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 应用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 俯视
        ocnt = 0
        # 平视
        xcnt = 0
        # 遍历每个轮廓
        for cnt in contours:
            # 计算轮廓的面积和周长
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            # 计算轮廓的长宽比
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            # 根据面积、周长和长宽比来判断物体的姿态
            if aspect_ratio > 0.85:
                ocnt = ocnt + 1
            else:
                xcnt = xcnt + 1
        return ocnt > xcnt

    def get_image_isoverlook_plus(self, img):
        '''
        判断是否为俯视拍摄的物体
        轮廓的面积和周长来进一步判断物体的姿态。
        例如，我们可以计算轮廓的紧密度（compactness），即周长的平方除以面积。
        紧密度可以反映轮廓的复杂程度，如果紧密度较大，可能说明物体是从上往下拍的，因为从上往下拍摄的物体轮廓通常更复杂。
        如果紧密度较小，可能说明物体是水平拍的，因为水平拍摄的物体轮廓通常更简单。
        '''

        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 应用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 俯视
        ocnt = 0
        # 平视
        xcnt = 0
        # 遍历每个轮廓
        for cnt in contours:
            # 计算轮廓的面积和周长
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            # 计算轮廓的紧密度
            compactness = perimeter ** 2 / area if area > 0 else 0

            # 根据紧密度来判断物体的姿态
            if compactness > 40:  # 这是一个经验值，可能需要根据你的数据进行调整
                ocnt = ocnt + 1
            else:
                xcnt = xcnt + 1
        return ocnt > xcnt

    def analyze_image(self, image: Image):
        # 将图像转换为 numpy 数组以便分析
        data = np.array(image)

        # 计算图像的主要颜色（这只是一个简单的示例，实际的颜色分析可能需要更复杂的算法）
        main_color = data.mean(axis=(0, 1))

        # 计算图像的形状（这只是一个简单的示例，实际的形状分析可能需要更复杂的算法）
        shape = data.shape[:2]

        return main_color, shape

    def add_gradient_effect(self, shadow_image: Image, output: str):
        width, height = shadow_image.size

        # 渐变效果处理
        for y in range(height):
            for x in range(width):
                # 计算距离图像右边的距离，生成渐变效果
                distance_to_right = width - x

                # 根据距离来计算渐变效果的颜色值
                gradient_value = distance_to_right / width
                new_opacity = int(255 * gradient_value)

                # 获取当前像素点的颜色并调整其透明度
                current_color = shadow_image.getpixel((x, y))
                new_color = current_color[:-1] + (new_opacity,)  # 保持原有颜色，只修改透明度

                # 将调整后的颜色应用到图像中
                shadow_image.putpixel((x, y), new_color)

        shadow_image.save(output)
        return shadow_image

    def check_image_orientation(self, image_path):
        '''
        检测图像的形状：竖图、横图
        :param image_path:
        :return:
        '''
        img = Image.open(image_path)
        if img.mode != "RGBA":
            return None
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        non_transparent = cv2.bitwise_and(img, img, mask=(img[:, :, 3] != 0).astype('uint8') * 255)
        gray = cv2.cvtColor(non_transparent, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        if w >= h:
            return 'w'
        else:
            return 'h'
