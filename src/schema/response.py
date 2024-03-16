from typing import List, Iterable

from pydantic import BaseModel

from database.models import DLModelDeployInDB
from schema.entity import DLModelEntity, DLModelVersionEntity, DLModelDeployEntity


class BaseResponse(BaseModel):
    code: int
    message: str = ""
    data: List = []


class DLModelEntityResponse(BaseResponse):
    data: List[DLModelEntity] = []


class DLModelVersionEntityResponse(BaseResponse):
    data: List[DLModelVersionEntity] = []


class DLModelDeployResponseItem(DLModelDeployEntity):
    model_name: str
    version: str

    @staticmethod
    def from_db_model(model: DLModelDeployInDB):
        return DLModelDeployResponseItem(
            id=model.id,
            create_time=model.create_time,
            user_id=model.user_id,
            version_id=model.version_id,
            display_name=model.display_name,
            description=model.description,
            model_name=model.model_version_item.model_item.register_name,
            version=model.model_version_item.version
        )


class DLModelDeployEntityResponse(BaseResponse):
    data: List[DLModelDeployResponseItem] = []


class ResponseFormatter:
    @staticmethod
    def success(data=None, message: str = "") -> BaseResponse:
        if data is None:
            data = []
        if not isinstance(data, Iterable):
            data = [data]
        return BaseResponse(code=200, message=message, data=data)

    @staticmethod
    def error(code: int = 500, message: str = "", data=None) -> BaseResponse:
        if data is None:
            data = []
        return BaseResponse(code=code, message=message, data=data)
