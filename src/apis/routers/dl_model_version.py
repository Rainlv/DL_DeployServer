from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from apis.services.mode_version_service import deploy_model_version, train_model
from database.db import get_session
from database.mapper import DLModelVersionMapper
from schema.request import TrainingArgsModel
from schema.response import DLModelVersionEntityResponse, ResponseFormatter, BaseResponse
from utils.constant import TrainingStatus

router = APIRouter(prefix="/model/version", tags=["模型版本"])


@router.get("", response_model=DLModelVersionEntityResponse)
def query_model_version(
        model_id: int,
        version: str | None = None,
        deploy_status: bool | None = None,
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    data = model_mapper.query(model_id, version, deploy_status)
    return ResponseFormatter.success(data=data)


@router.post("/{version_id}", description="部署模型至工具箱", response_model=BaseResponse)
def deploy_model(
        version_id: int,
        display_name: str,
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    model_version_obj = model_mapper.get_by_id(version_id)
    if not model_version_obj:
        return ResponseFormatter.error(code=404, message="模型版本不存在")
    if model_version_obj.deploy_status:
        return ResponseFormatter.error(code=400, message="模型版本已部署")
    if deploy_model_version(model_version_obj):
        model_mapper.deploy(model_version_obj, display_name)
        return ResponseFormatter.success(message="部署成功")
    return ResponseFormatter.error(code=500, message="部署失败")


@router.put("/{version_id}", description="编辑模型版本", response_model=BaseResponse)
def update_model_version(
        version_id: int,
        train_status: int = None,
        description: str = None,
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    model_version_obj = model_mapper.get_by_id(version_id)
    if not model_version_obj:
        return ResponseFormatter.error(code=404, message="模型版本不存在")
    model_version_obj = model_mapper.edit(model_version_obj, train_status, description)
    return ResponseFormatter.success(message="更新成功")


@router.put("/{version_id}/train", description="启动训练模型", response_model=BaseResponse)
def train_model_api(
        version_id: int,
        training_arg: TrainingArgsModel,
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    model_version_obj = model_mapper.get_by_id(version_id)
    if not model_version_obj:
        return ResponseFormatter.error(code=404, message="模型版本不存在")
    if model_version_obj.train_status in [TrainingStatus.TRAINING.value, TrainingStatus.PREPARING.value]:
        return ResponseFormatter.error(code=400, message="模型训练中")
    elif model_version_obj.train_status == TrainingStatus.SUCCESS.value:
        return ResponseFormatter.error(code=400, message="模型训练已完成")
    if train_model(model_version_obj, training_arg):
        model_mapper.edit(model_version_obj, train_status=TrainingStatus.TRAINING.value)
        return ResponseFormatter.success(message="启动训练任务成功")
    return ResponseFormatter.error(code=501, message="启动训练任务失败")


@router.put("/inner/{version_id}/train_done/{training_status}", description="模型训练完成回调",
            response_model=BaseResponse)
def __train_done_callback(
        version_id: int,
        training_status: int,
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    model_version_obj = model_mapper.get_by_id(version_id)
    if not model_version_obj:
        return ResponseFormatter.error(code=404, message="模型版本不存在")
    model_version_obj = model_mapper.edit(model_version_obj, train_status=training_status)
    return ResponseFormatter.success(message="更新成功")
