from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from apis.services.mode_version_service import deploy_model_version
from database.db import get_session
from database.mapper import DLModelDeployMapper
from schema.response import ResponseFormatter

router = APIRouter(prefix="/model/deploy", tags=["工具箱"])


@router.get("", description="获取已部署的模型工具箱列表")
def get_model_deploy(db: AsyncSession = Depends(get_session)):
    model_mapper = DLModelDeployMapper(db)
    data = model_mapper.list()
    return ResponseFormatter.success(data=data)
