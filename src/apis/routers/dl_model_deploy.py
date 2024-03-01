from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_session
from database.mapper import DLModelDeployMapper
from schema.response import ResponseFormatter

route = APIRouter(prefix="/model/deploy")


@route.get("", description="获取已部署的模型工具箱列表")
async def get_model_deploy(db: AsyncSession = Depends(get_session)):
    model_mapper = DLModelDeployMapper(db)
    data = model_mapper.list()
    return ResponseFormatter.success(data=data)
