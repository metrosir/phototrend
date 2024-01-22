from api.base import ApiBase
from api.functions import *

import asyncio
import glob


class ImageBgGenerateSyncResultV1(ApiBase):

    def params_data(self):
        if self.query_params['id_task'] is None:
            raise Exception("params error")
        return self.query_params

    async def action(self):
        if 'idx' not in self.query_params:
            img_list = glob.glob(datadir.api_generate_commodity_dir.format(id_task=self.params['id_task'],
                                                                           type="output", ) + "/*.png")
        else:
            img_list = glob.glob(datadir.api_generate_commodity_dir.format(id_task=self.params['id_task'],
                                                                           type="output", ) + f"/{int(self.params['idx'])}.png")
        if len(img_list) < 1:
            return []

        async def encode_image_to_base64(path):
            def sync_encode_image_to_base64(path):
                with Image.open(path) as im:
                    if im.size == (0, 0):
                        return None
                    im = im.convert("RGB")
                    return encode_to_base64(im)
            return await asyncio.to_thread(sync_encode_image_to_base64, path)

        async def cys():
            tasks = [encode_image_to_base64(path) for path in img_list]
            res = await asyncio.gather(*tasks)
            return res

        data = await cys()
        return data
