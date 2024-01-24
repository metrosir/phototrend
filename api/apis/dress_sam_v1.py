from api.base import ApiBase
from api.functions import *

import utils.datadir as datadir
import uuid
import os
import pathlib
import datetime
import asyncio

from scripts.dress.sam import Sam
from scripts.dress.pose import Pose
from scripts.dress.rembg import Rembg


class DressSamV1(ApiBase):

        def params_data(self):
            return {}

        async def action(self):
            res = {
                "pose": False,
                "distance": None,
                "sam": [],
                "rembg": ""
            }
            form_data = await self.request.form()
            file = form_data['file']
            self.params['uid'] = form_data['uid']
            self.params['id_task'] = form_data['id_task']

            content = await file.read()
            dir = datadir.dress_worker_history.format(worker_id=self.params['id_task'], type='input')
            dir = os.path.join(dir, 'input')
            if not os.path.exists(dir):
                pathlib.Path.mkdir(pathlib.Path(dir), parents=True, exist_ok=True)
            file_path = os.path.join(dir, 'dress.' + file.filename.split('.')[-1])
            with open(file_path, 'wb') as f:
                f.write(content)
                f.close()

            pose_obj = Pose(file_path)
            res['pose'], res['distance'] = pose_obj.isNotPerson()
            pose_obj.getPose().save(dir + '/pose.png')

            if not res['pose']:
                return res
            # return res
            # sam
            sam = Sam(file_path)
            # rem
            rembg = Rembg(file_path, dir + '/rembg.png')

            # res['sam'] = sam()
            # rembg_img_path = rembg()

            res['sam'], res['rembg'] = await asyncio.gather(sam(), rembg(self.params['uid']), self.params['id_task'])
            return res







