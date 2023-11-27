import math

from PIL import Image
from PIL import ImageEnhance


from utils import image as util


def calc_color_temperature(temp):
	white = (255.0, 254.11008387561782, 250.0419083427406)

	temperature = temp / 100

	if temperature <= 66:
		red = 255.0
	else:
		red = float(temperature - 60)
		red = 329.698727446 * math.pow(red, -0.1332047592)
		if red < 0:
			red = 0
		if red > 255:
			red = 255

	if temperature <= 66:
		green = temperature
		green = 99.4708025861 * math.log(green) - 161.1195681661
	else:
		green = float(temperature - 60)
		green = 288.1221695283 * math.pow(green, -0.0755148492)
	if green < 0:
		green = 0
	if green > 255:
		green = 255

	if temperature >= 66:
		blue = 255.0
	else:
		if temperature <= 19:
			blue = 0.0
		else:
			blue = float(temperature - 10)
			blue = 138.5177312231 * math.log(blue) - 305.0447927307
			if blue < 0:
				blue = 0
			if blue > 255:
				blue = 255

	return red / white[0], green / white[1], blue / white[2]


class FinalProcessorBasic:
	def __init__(self, params) -> None:
		# 对比度
		self.contrast = params['contrast']
		# 亮度
		self.brightness = params['brightness']
		# 锐度
		self.sharpeness = params['sharpeness']
		# 饱和度
		self.color_saturation = params['color_saturation']
		# 色温
		self.color_temperature = params['color_temperature']
		# 噪点
		self.noise_alpha_final = params['noise_alpha_final']

	def preprocess(self, params):
		return True

	def process(self, seed, image: Image):

		if self.noise_alpha_final != 0:
			img_noise = util.generate_noise(seed, image.size[0], image.size[1])
			image = Image.blend(image, img_noise, alpha=self.noise_alpha_final)

		if self.contrast != 1:
			enhancer = ImageEnhance.Contrast(image)
			image = enhancer.enhance(self.contrast)

		if self.brightness != 1:
			enhancer = ImageEnhance.Brightness(image)
			image = enhancer.enhance(self.brightness)

		if self.sharpeness != 1:
			enhancer = ImageEnhance.Sharpness(image)
			image = enhancer.enhance(self.sharpeness)

		if self.color_saturation != 1:
			enhancer = ImageEnhance.Color(image)
			image = enhancer.enhance(self.color_saturation)

		if self.color_temperature != 0:
			temp = calc_color_temperature(6500 + self.color_temperature)
			az = []
			for d in image.getdata():
				az.append((int(d[0] * temp[0]), int(d[1] * temp[1]), int(d[2] * temp[2])))
			image = Image.new('RGB', image.size)
			image.putdata(az)

		return image