import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeployInfo:
    serialized_file_path: str
    model_name: str
    version: str
    handler_path: str
    extra_files: str


class DeployEngine:
    def __init__(self):
        self.model_store_dir: Path = None

    def deploy(self, deploy_info: DeployInfo):
        cmd = f"torch-model-archiver  \
              -v {deploy_info.version}  \
              --serialized-file {deploy_info.serialized_file_path} \
              --model-name {deploy_info.model_name} \
              --handler {deploy_info.handler_path} \
              --extra-files {deploy_info.extra_files} \
              --export-path {self.model_store_dir}"
        subprocess.run(cmd)
