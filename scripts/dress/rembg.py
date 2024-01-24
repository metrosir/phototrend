from utils.constant import catout_apiket, rembg_api
from utils.s3 import upload as s3_upload
import os


class Rembg:
    def __init__(self, input_img, out_img, is_call_api=True):
        self.input_img = input_img
        self.out_img = out_img
        self.is_call_api = is_call_api

    async def __call__(self, cloud_uid, worker_id):
        if self.is_call_api:
            await self.call_api()
        else:
            self.call_loca()

        if os.path.exists(self.out_img):
            # todo s3
            img_url = s3_upload(self.out_img, cloud_uid, worker_id)
            return img_url
        return None

    def call_loca(self):
        return False

    async def call_api(self):
        from utils.req import async_req_base
        response = await async_req_base(url=rembg_api, method="POST", data={'file': open(self.input_img, 'rb')}, heades={'APIKEY': catout_apiket})
        with open(self.out_img, 'wb') as out:
            out.write(response)
        return True
