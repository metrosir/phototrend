import requests
import sys
import json
import os
import shutil
import cv2
import numpy as np

from utils.constant import sam_host


class Sam:

    def __init__(self, img_path, is_call_api=True):
        self.img_path = img_path
        self.is_call_api = is_call_api

    async def __call__(self, *args, **kwargs):
        if self.is_call_api:
            result = await self.call_api()
            mask_list = result['list']
        else:
            mask_list = self.call_loca()
        return mask_list

    def preinstall(self):
        pass

    def call_loca(self):
        return False

    async def call_api(self):
        from utils.req import async_req_base
        from aiohttp import FormData

        # Todo
        url = f"{sam_host}/seg_upload"
        # files = {'file': open(self.img_path, "rb")}
        # data = {
        #     "file_type": 0,
        #     "image_url": ""
        # }
        # data = {
        #     'file': open(self.img_path, "rb"),
        #     "file_type": "0",
        #     "image_url": ""
        # }

        r_result = None
        try:
            data = FormData()
            data.add_field('file', open(self.img_path, "rb"))
            data.add_field("file_type", "0")
            data.add_field("image_url", "")
            ret = await async_req_base(url=url, method="POST", data=data, heades={})
            status = ret['status']
        except Exception as e:
            raise Exception(f"Call sam api error: {str(e)}. result: {r_result}") if r_result is not None else Exception(f"Call sam api error: {str(e)}")
        if int(status) == 200:
            return ret['data']
        else:
            raise Exception(f"Call sam api error: {str(ret)}")
