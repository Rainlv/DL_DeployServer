from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_session
from database.mapper import DLModelMapper
from schema.response import DLModelEntityResponse, ResponseFormatter

router = APIRouter(prefix="/model", tags=["模型管理"])


@router.get("", response_model=DLModelEntityResponse)
async def query_model(
        name: str | None = None,
        task_type_name: str | None = None,
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelMapper(db)
    data = model_mapper.query(name, task_type_name)
    return ResponseFormatter.success(data=data)
