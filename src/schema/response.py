from typing import Any, List

from pydantic import BaseModel

from schema.entity import DLModelEntity, DLModelVersionEntity, DLModelDeployEntity


class BaseResponse(BaseModel):
    code: int
    message: str = ""
    data: List = []


class DLModelEntityResponse(BaseResponse):
    data: List[DLModelEntity] = []


class DLModelVersionEntityResponse(BaseResponse):
    data: List[DLModelVersionEntity] = []


class DLModelDeployEntityResponse(BaseResponse):
    data: List[DLModelDeployEntity] = []


class ResponseFormatter:
    @staticmethod
    def success(data: Any = None, message: str = "") -> BaseResponse:
        return BaseResponse(code=200, message=message, data=data)

    @staticmethod
    def error(code: int = 500, message: str = "") -> BaseResponse:
        return BaseResponse(code=code, message=message)
