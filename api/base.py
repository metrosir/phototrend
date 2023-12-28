from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field
from fastapi import Request, Depends
import asyncio
import time

from utils.pt_logging import ia_logging, log_echo


class ApiResponse(BaseModel):
    data: dict = Field(..., description="response data")
    status: int = Field(..., description="response status")
    message: str = Field(..., description="response message")
    duration: float = Field(..., description="response duration")


class ApiBase(ABC):
    params = None
    request = None
    query_params = {}

    def __init__(self):
        pass
        # asyncio.run(self.__call__(request=request))
        # asyncio.create_task(self.__call__(req=req))
        # self.duration = round(time.time() - start_time, 5)

    async def __call__(self, request: Request):
        self.data = []
        self.status = 200
        self.message = 'success'
        self.duration = 0
        self.start_time = time.time()

        transfer_duration = 0
        try:
            if request.method == "POST":
                self.request = await request.json()
                transfer_duration = float(round(time.time() - self.start_time, 5))
            if request.query_params.items() is not None:
                for k, v in request.query_params.items():
                    self.query_params[k] = v
            self.before()
            self.params = self.params_data()
            self.data = await self.action()

            self.duration = float(round(time.time() - self.start_time, 5))
        except Exception as e:
            self.status = 500
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
            return {
                "data": self.data,
                "status": self.status,
                "message": self.message,
                "duration": self.duration,
            }

    @abstractmethod
    async def action(self):
        pass

    @abstractmethod
    def params_data(self):
        pass

    def before(self):
        pass

    def after(self, req: Request, **kwargs):
        msg = {
            "status": self.status,
            "message": self.message,
            "duration": self.duration,
            "api": req.url.path,
            "client_host": req.client.host,
            "host": req.headers['host'],
            "req_params": str(req.query_params),
            "req_body": self.request,
        }
        for k, v in kwargs.items():
            msg[k] = v
        log_echo(title="api info", exception=None, msg=msg, level="info", path=req.url.path)
