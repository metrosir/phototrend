from api.base import ApiBase
from api.functions import *

import utils.datadir as datadir
import uuid
import os
import pathlib
import datetime
import asyncio

from api.pipe_tasks.base import InputWorkerData, gtype_dress

from api.functions import G_PIPE


class DressRunV1(ApiBase):

    def params_data(self):
        return self.params

    async def action(self):
        await InputWorkerData(self.request, G_PIPE, None).action(gtype=gtype_dress)
        return []
