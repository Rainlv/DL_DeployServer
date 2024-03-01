from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_session
from database.mapper import DLModelVersionMapper
from schema.response import DLModelVersionEntityResponse, ResponseFormatter

route = APIRouter(prefix="/model/version")


@route.get("", response_model=DLModelVersionEntityResponse)
async def query_model_version(
        model_id: int,
        version: str | None = None,
        deploy_status: bool | None = None,
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    data = model_mapper.query(model_id, version, deploy_status)
    return ResponseFormatter.success(data=data)
