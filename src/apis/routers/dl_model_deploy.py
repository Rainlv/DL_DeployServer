from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_session
from ..db_crud.dl_model_deploy import DLModelDeployMapper

route = APIRouter(prefix="/model/deploy")


@route.get("")
async def get_model_deploy(db: AsyncSession = Depends(get_session)):
    model_mapper = DLModelDeployMapper(db)
    return model_mapper.list()
