from fastapi import APIRouter
from .routers.dl_model import router as dl_model_router
from .routers.dl_model_deploy import router as dl_model_deploy_router
from .routers.dl_model_version import router as dl_model_version_router

root_router = APIRouter(prefix="/api")

root_router.include_router(dl_model_router)
root_router.include_router(dl_model_deploy_router)
root_router.include_router(dl_model_version_router)

