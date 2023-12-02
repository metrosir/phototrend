
import os
import threading
import utils.pt_logging as pt_logging


class Imageshadowss:
    def __init__(self, x_offset, y_offset, blur, opacity, toggle=0, color="#000000"):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.blur = blur
        self.color = color
        self.opacity = opacity
        self.toggle = toggle

    def __call__(self, img_input_path, img_output_path):
        try:
            lock = threading.Lock()
            lock.acquire()
            # infile outfile x y blur color opacity toggle
            cmd=f"gimp -i -b '(add-shadow \"{img_input_path}\" \"{img_output_path}\" {self.x_offset} {self.y_offset} {self.blur} \"{self.color}\" {self.opacity} {self.toggle})' -b '(gimp-quit 0)'"
            print(cmd)
            os.system(cmd)
            lock.release()
        except Exception as e:
            pt_logging.log_echo(
                title="add-shadow error",
                msg={"img_input_path": img_input_path, "img_output_path": img_output_path},
                exception=e,
                level="error",
            )
            return False
        finally:
            pass
        return True