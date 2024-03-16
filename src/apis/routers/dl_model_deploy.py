from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_session
from database.mapper import DLModelDeployMapper
from schema.response import ResponseFormatter, DLModelDeployEntityResponse, DLModelDeployResponseItem

router = APIRouter(prefix="/model/deploy", tags=["工具箱"])


@router.get("", description="获取已部署的模型工具箱列表", response_model=DLModelDeployEntityResponse)
def get_model_deploy(db: AsyncSession = Depends(get_session)):
    model_mapper = DLModelDeployMapper(db)
    data = model_mapper.list()
    resp_data = [DLModelDeployResponseItem.from_db_model(item) for item in data]
    return ResponseFormatter.success(data=resp_data)
