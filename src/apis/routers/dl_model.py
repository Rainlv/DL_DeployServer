from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_session
from database.mapper import DLModelMapper
from schema.response import DLModelEntityResponse, ResponseFormatter

router = APIRouter(prefix="/model", tags=["模型管理"])


@router.get("", response_model=DLModelEntityResponse, description="查询与筛选模型")
async def query_model(
        name: str | None = Query(None, title="模型名称", description="模型名称", max_length=100),
        task_type_name: str | None = Query(None, title="任务类型名称", description="任务类型名称",
                                           max_length=100),
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelMapper(db)
    data = model_mapper.query(name, task_type_name)
    return ResponseFormatter.success(data=data)
