import pathlib

from api.base import ApiBase
from api.functions import *
from utils.datadir import project_dir
import uuid
from fastapi.responses import FileResponse
from fastapi import Request, Depends


class CollageImageTemplate(ApiBase):

    def params_data(self):
        return self.params

    async def action(self):
        # self.query_params['id_task']
        if self.query_params['type'] == 'temp':
            dir = os.path.join(project_dir, 'worker_data', 'collage_template_file', 'templates')
            return FileResponse(os.path.join(dir, self.query_params['name']))
        # elif self.params['type'] == 'list':
            # type_file = os.path.join(project_dir, 'worker_data', 'collage_template_file', 'list.json')
        return FileResponse(os.path.join(project_dir, 'worker_data', 'collage_template_file', 'list.json'))


    async def __call__(self, request: Request):
        self.data = []
        self.status = 200
        self.message = 'success'
        self.duration = 0
        self.start_time = time.time()

        transfer_duration = 0
        try:
            if request.method == "POST":
                # 判断是否为文件上传
                if not request.headers['content-type'].startswith('multipart/form-data'):
                #     self.request = await request.body()
                # else:
                    self.request = await request.form()
                else:
                    self.request = request
                transfer_duration = float(round(time.time() - self.start_time, 5))
            if request.query_params.items() is not None:
                self.query_params = {}
                for k, v in request.query_params.items():
                    self.query_params[k] = v
            self.before()
            self.params = self.params_data()
            self.data = await self.action()

            self.duration = float(round(time.time() - self.start_time, 5))
        except Exception as e:
            self.status = 400
            self.message = "error"
            log_echo(title="api error", exception=e, msg={
                "api": request.url.path,
                "client_host": request.client.host,
                "host": request.headers['host'],
                "req_params": str(request.query_params),
                "req_body": self.request,
                "transfer_duration": transfer_duration,
            }, level="error", is_collect=True, path=request.url.path)
        finally:
            self.after(request, transfer_duration=transfer_duration)
            # response_fields = ApiResponse.__annotations__.keys()
            # attributes = dir(self)
            # for attr in attributes:
            #     if attr not in response_fields and not attr.startswith('__'):
            #         try:
            #             delattr(self, attr)
            #         except:
            #             pass
            return self.data