import pathlib

from api.base import ApiBase
from api.functions import *
from utils.datadir import project_dir
import uuid
from utils.s3 import upload


class ImageFileUpload(ApiBase):

    def params_data(self):
        return self.params

    async def action(self):
        form_data = await self.request.form()
        file = form_data['file']



        # type_value = form_data['type_value']
        # type_label = form_data['type_label']
        # label = form_data['label']
        # value = form_data['value']
        tname = f'{uuid.uuid4().hex[:8]}.png'
        dir = os.path.join(project_dir, 'worker_data', 'collage_template_file', 'templates')
        if not os.path.exists(dir):
            pathlib.Path.mkdir(pathlib.Path(dir), parents=True, exist_ok=True)
        with open(os.path.join(dir, tname), 'wb') as f:
            f.write(await file.read())
            f.close()
        return upload(os.path.join(dir, tname), 'collage_template_file')

        # types_dir = os.path.join(project_dir, 'worker_data', 'collage_template_file')
        # types_name = f'list.json'
        # list_file = os.path.join(types_dir, types_name)
        # # if not os.path.exists(list_file):
        # #     with open(list_file, 'w') as f:
        # #         json.dump(["data"], f)
        # #         f.close()
        # with open(list_file, 'r') as ff:
        #     data = json.load(ff)
        #     ff.close()
        # res = {
        #     "value": type_value,
        #     "label": type_label,
        #     "list": [
        #         {
        #             "label": label,
        #             "value": value,
        #             "tempUrl": f"http://10.61.158.18:7777/v1/temp_file?type=temp&name={tname}",
        #             "src": "http://10.61.158.18:7777/read_img?path=/data1/aigc/vue-fabric-editor/data/template/type-1-1.png"
        #         }
        #     ]
        # }
        #
        # if len(data['data']) < 1:
        #     data['data'] = [res]
        # else:
        #     if type_value not in [d['value'] for d in data['data']]:
        #         data['data'].append(res)
        #     else:
        #         for d in data['data']:
        #             if d['value'] == type_value:
        #                 d['list'].append(res['list'][0])
        #
        # with open(list_file, 'w') as f:
        #     f.write(json.dumps(data))
        #     f.close()