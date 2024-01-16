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


class DressSam(ApiBase):

        def params_data(self):
            return {
                "id_task": self.request.get("id_task", datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4]),
                "uid": self.request.get("uid", uuid.uuid4().hex[:8]),
            }

        async def action(self):
            res = {
                "pose": False,
                "distance": None,
                "sam": [],
                "rembg": ""
            }
            form_data = await self.request.form()
            file = form_data['file']
            content = await file.read()
            id_task = self.params['id_task']
            dir = datadir.dress_worker_history.format(worker_id=id_task)
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

            res['sam'], res['rembg'] = await asyncio.gather(sam(), rembg(self.params['uid']))
            return res







