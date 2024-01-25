from api.base import ApiBase
from api.functions import G_PIPE, interrogate, queue
from api.functions import saveimage
import json
import asyncio

from api.pipe_tasks.base import InputWorkerData, gtype_commodity


class ImageBgGenerateSyncV1(ApiBase):

    def params_data(self):
        if self.request['id_task'] is None or len(self.request['data']['input_images']) < 1 or self.request['data']['mask_image'] is None:
            raise Exception("params error")
        return self.params

    async def action(self):
        await InputWorkerData(self.request, G_PIPE, interrogate, gtype=gtype_commodity).action()
        return []

        data = self.request
        saveimage(
            id_task=data['id_task'],
            _type="input",
            images=[data['data']['input_images'][0], data['data']['mask_image']]
        )

        data['data']['input_images'] = []
        data['data']['mask_image'] = ''
        # asyncio.create_task(queue.enqueue(json.dumps(data)))
        queue.enqueue(json.dumps(data))
        return []
