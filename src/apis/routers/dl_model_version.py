import uuid
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, UploadFile, File, Query, Path
from fastapi.params import Body
from sqlalchemy.ext.asyncio import AsyncSession

from apis.services.mode_version_service import deploy_model_version, train_model
from apis.services.model_deploy_service import DeployEngine, DeployInfo
from config import config
from database.db import get_session
from database.mapper import DLModelVersionMapper
from schema.request import TrainingArgsModel, CreateVersionRequestModel, CreateDeployRequestModel
from schema.response import DLModelVersionEntityResponse, ResponseFormatter, BaseResponse
from utils.constant import TrainingStatus

router = APIRouter(prefix="/model/version", tags=["模型版本"])


@router.get("",
            response_model=DLModelVersionEntityResponse,
            description="查询与筛选模型版本")
def query_model_version(
        model_id: int = Query(..., title="模型ID", description="模型ID"),
        version: str | None = Query(None, title="模型版本", description="模型版本"),
        deploy_status: bool | None = Query(None, title="部署状态", description="部署状态 true已部署 false未部署"),
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    data = model_mapper.query(model_id, version, deploy_status)
    return ResponseFormatter.success(data=data)


@router.post("", response_model=DLModelVersionEntityResponse, description="创建模型版本")
def create_version(
        version_obj: CreateVersionRequestModel,
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    model_version_obj = model_mapper.create(
        model_id=version_obj.model_id,
        sample_set_id=version_obj.sample_set_id,
        description=version_obj.description,
        version=version_obj.version
    )
    return ResponseFormatter.success(data=model_version_obj)


@router.post("/{version_id}", description="部署模型至工具箱", response_model=BaseResponse)
def deploy_model(
        version_id: int = Path(..., title="模型版本ID", description="模型版本ID"),
        display_name: CreateDeployRequestModel = Body(..., title="工具箱显示名称", description="工具箱显示名称"),
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    model_version_obj = model_mapper.get_by_id(version_id)
    if not model_version_obj:
        return ResponseFormatter.error(code=404, message="模型版本不存在")
    if model_version_obj.deploy_status:
        return ResponseFormatter.error(code=400, message="模型版本已部署")
    if deploy_model_version(model_version_obj):
        model_mapper.deploy(model_version_obj, display_name.display_name)
        return ResponseFormatter.success(message="部署成功")
    return ResponseFormatter.error(code=500, message="部署失败")


@router.put("/{version_id}", description="编辑模型版本", response_model=BaseResponse)
def update_model_version(
        version_id: int = Path(..., title="模型版本ID", description="模型版本ID"),
        train_status: int = Body(None, title="训练状态",
                                 description="训练状态 0训练完成 1未开始 -1训练失败 2准备中 3训练中"),
        description: str = Body(None, title="模型版本描述", description="模型版本描述"),
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
        version_id: int = Path(..., title="模型版本ID", description="模型版本ID"),
        training_arg: TrainingArgsModel = Body(..., title="训练参数", description="训练参数"),
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


@router.delete("/{version_id}", description="删除模型版本", response_model=BaseResponse)
def delete_model_version(
        version_id: int = Path(..., title="模型版本ID", description="模型版本ID"),
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    if model_mapper.delete(version_id):
        return ResponseFormatter.success(message="删除成功")
    else:
        return ResponseFormatter.error(code=404, message="模型版本不存在")


@router.post("/inner/{version_id}/train_done", description="内部接口，模型训练完成回调",
             response_model=BaseResponse)
async def __train_done_callback(
        version_id: int,
        file: UploadFile = File(...),
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    model_version_obj = model_mapper.get_by_id(version_id)
    if not model_version_obj:
        return ResponseFormatter.error(code=404, message="模型版本不存在")

    try:
        torch_script_file = NamedTemporaryFile(suffix=".torchscript", delete=False)
        torch_script_file.write(await file.read())
        model_file_name = uuid.uuid4().hex  # model_name为文件名，不能重复，具体model_name在发布时指定
        DeployEngine.deploy(DeployInfo(
            model_name=model_file_name,
            serialized_file_path=torch_script_file.name,
            version=model_version_obj.version,
            handler_path=f"{config.MAR_DEPS_DIR}/handlers/{model_version_obj.model_item.register_name}_handler.py",
            extra_files=f"{config.MAR_DEPS_DIR}/extra_files",
        ))

        model_mapper.edit(model_version_obj, train_status=TrainingStatus.SUCCESS.value,
                          mar_file_path=f"{model_file_name}.mar")
    except Exception as e:
        model_mapper.edit(model_version_obj, train_status=TrainingStatus.FAILED.value)
        return ResponseFormatter.error(code=500, message=f"部署失败: {e}")
    return ResponseFormatter.success(message="更新成功")


@router.post("/inner/{version_id}/train_fail", description="内部接口，模型训练失败回调",
             response_model=BaseResponse)
async def __train_fail_callback(
        version_id: int,
        db: AsyncSession = Depends(get_session)
):
    model_mapper = DLModelVersionMapper(db)
    model_version_obj = model_mapper.get_by_id(version_id)
    if not model_version_obj:
        return ResponseFormatter.error(code=404, message="模型版本不存在")
    model_mapper.edit(model_version_obj, train_status=TrainingStatus.FAILED.value)
    return ResponseFormatter.success(message="更新成功")
