from api.base import ApiBase
from api.functions import *

import utils.datadir as datadir
import uuid
import os
import pathlib
import datetime
import asyncio

# from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from api.pipe_tasks.input_worker_data import InputWorkerData


class DressRunV1(ApiBase):

    def params_data(self):
        return self.params

    async def action(self):
        await InputWorkerData(self.request, None, None).action()

        # pass
        # todo 换装处理步骤
        # 1.模特图和场景图进行贴合
        # 2.模特图和场景图进行融合


# 贴合时可能出现的情况：
# 第一类情况，大小不一致
# 1）模特图大于场景图
# 2）模特图小于场景图
# 3）模特图和场景图大小一致
#
# 第二类情况，模特图和场景图的位置不一致
# 1）模特图在场景图的左上角
# 2）模特图在场景图的左下角
# 3）模特图在场景图的右上角
# 4）模特图在场景图的右下角
# 5）模特图在场景图的中间
#
#
# 第三类情况，模特图和场景图的角度不一致
# 第四类情况，模特图和场景图的光照不一致
# 第五类情况，模特图和场景图的背景不一致
# 第六类情况，模特图和场景图的分辨率不一致
# 第七类情况，模特图和场景图的色彩不一致
# 第八类情况，模特图和场景图的纹理不一致
# 第九类情况，模特图和场景图的风格不一致
