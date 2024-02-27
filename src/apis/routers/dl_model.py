from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from apis.db_crud.dl_model import DLModelMapper
from database.db import get_session
from schema.entity import DLModelEntity

route = APIRouter(prefix="/model")


@route.get("", response_model=list[DLModelEntity])
async def get_model(db: AsyncSession = Depends(get_session)):
    model_mapper = DLModelMapper(db)
    return model_mapper.list()


@route.post("")
async def create_model():
    pass
