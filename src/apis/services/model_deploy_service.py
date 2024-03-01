import subprocess
from dataclasses import dataclass

from loguru import logger

from config import config


@dataclass
class DeployInfo:
    serialized_file_path: str
    version: str
    handler_path: str
    extra_files: str
    model_name: str = ""


class DeployEngine:
    @staticmethod
    def deploy(deploy_info: DeployInfo):
        cmd = f"torch-model-archiver  \
              -v {deploy_info.version}  \
              --serialized-file {deploy_info.serialized_file_path} \
              --model-name {deploy_info.model_name} \
              --handler {deploy_info.handler_path} \
              --extra-files {deploy_info.extra_files} \
              --export-path {config.MODEL_STORE_DIR}"
        res = subprocess.run(cmd)
        if res.returncode == 0:
            return True
        logger.error(f"Failed to deploy model: {deploy_info}, details: {res.stdout}")
        return False
