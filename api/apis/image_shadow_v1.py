from api.base import ApiBase
from scripts.gimpscripts.shadow import Imageshadowss,ImagePerspectiveShadow
from utils.constant import shadow
from utils.constant import project_dir
from utils.pt_logging import ia_logging
from utils.image import decode_base64_to_image

import pathlib
import os
from fastapi import Request
import asyncio


class ImageShadowV1(ApiBase):

    # async def __call__(self, request: Request):
    #     await self.__call__(request=request)

    def params_data(self):
        self.params = {
            "id_task": self.request.get("id_task", ""),
            "app_id": self.request.get("app_id", ""),
            "user_id": self.request.get("user_id", ""),
            "data": {
                "callback_url": self.request.get("data", {}).get("callback_url", ""),
                "input_images": self.request.get("data", {}).get("input_images", []),
                # plane, perspective
                "handle_type": self.request.get("data", {}).get("handle_type", "plane"),
                # left, right
                "target": self.request.get("data", {}).get("target", "left"),
            },
        }
        return self.params

    async def action(self):
        ia_logging.info(f"ImageShadowV1 id_task: {self.params['id_task']}")

        input_path = os.path.join(project_dir, f"worker_data/history/simple_color_commodity/{self.params['id_task']}/input/")
        output_path = os.path.join(project_dir, f"worker_data/history/simple_color_commodity/{self.params['id_task']}/output/")
        pathlib.Path(input_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        input_path = os.path.join(input_path, f"0.png")
        output_path = os.path.join(output_path, f"0.png")

        decode_base64_to_image(encoding=self.params["data"]['input_images'][0], convert="RGBA", save_path=input_path)
        if self.params["data"]['handle_type'] == 'plane':
            im_shadow = Imageshadowss(
                shadow['plane'][self.params['data']['target']]['x'],
                shadow['plane'][self.params['data']['target']]['y'],
                shadow['plane'][self.params['data']['target']]['blur'],
                shadow['plane'][self.params['data']['target']]['opacity'],
                bg_color=shadow['plane'][self.params['data']['target']]['bg_color']
            )

        else:
            im_shadow = ImagePerspectiveShadow(
                shadow['perspective'][self.params['data']['target']]['angle'],
                shadow['perspective'][self.params['data']['target']]['distance'],
                shadow['perspective'][self.params['data']['target']]['length'],
                shadow['perspective'][self.params['data']['target']]['blur'],
                shadow['perspective'][self.params['data']['target']]['opacity'],
                shadow['perspective'][self.params['data']['target']]['bg_color'],
                shadow['perspective'][self.params['data']['target']]['gradient'],
                shadow['perspective'][self.params['data']['target']]['interpolation'],
                shadow['perspective'][self.params['data']['target']]['allow_resize'],
            )
        im_data = im_shadow(input_path, output_path, True)
        return [
            im_data
        ]

