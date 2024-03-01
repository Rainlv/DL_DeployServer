import os
import uuid
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch
from osgeo import gdal
from osgeo.gdal import Dataset, Driver
from ts.torch_handler.base_handler import BaseHandler

from data_types import InferenceContextGenerator, InferenceContext, CommonRequest
from notify import NotifierClient


def get_suffix_from_driver(driver):
    if driver == 'GTiff':
        return '.tif'
    elif driver == 'JPEG':
        return '.jpg'
    elif driver == 'PNG':
        return '.png'
    return ''


class CommonS3Handler(BaseHandler):
    RESULT_OSS_PREFIX = "inference_result/"

    def __init__(self, out_band_num=3, image_size=512, driver="PNG"):
        super(CommonS3Handler, self).__init__()
        os.environ["CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE"] = "YES"  # for GDAL to allow writing GTiff
        os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"  # for GDAL to resolve endpoint correctly
        os.environ["AWS_HTTPS"] = "FALSE"
        os.environ["AWS_DEFAULT_REGION"] = ""
        os.environ["AWS_SESSION_TOKEN"] = ""
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("MINIO_ACCESS_KEY")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MINIO_SECRET_KEY")
        os.environ["AWS_S3_ENDPOINT"] = os.environ.get("MINIO_ENDPOINT")

        self.out_band_num = out_band_num
        self.image_size = image_size
        self.driver = driver

        self.notifier_client = NotifierClient()

    def initialize(self, context):
        self.properties = context.system_properties
        self.manifest = context.manifest
        self._get_device_and_map_location()

        model_dir = self.properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        self.model_pt_path = os.path.join(model_dir, serialized_file)
        self._load_model()

        self.initialized = True

    def parse_request(self, requests) -> Any:
        request = requests[0]
        return self._parse_request(request)

    def preprocess(self, req: CommonRequest) -> InferenceContextGenerator:
        """Preprocess request input"""
        return self._preprocess_one_request(req)

    def inference(self, data: InferenceContextGenerator, *args, **kwargs):
        """Inference on data"""
        for ctx in data:
            ctx.output_data = self._inference_one_block(ctx)
            yield ctx

    def _inference_one_block(self, ctx: InferenceContext):
        return self.model(ctx.preprocessed_data)

    def postprocess(self, data: InferenceContextGenerator, *args, **kwargs):
        """Postprocess inference output"""
        ctx_ = None
        for ctx in data:
            ctx_ = ctx
            final_output = self._post_process_one_image(ctx).astype(np.uint8)
            ctx.out_ds.WriteArray(final_output, ctx.x_off, ctx.y_off, band_list=range(1, self.out_band_num + 1))
        final_ds = self._get_output_gdal_dataset(ctx_)
        resp = self._format_response(final_ds)
        self.clean(ctx_)
        return resp

    def clean(self, ctx: InferenceContext):
        drv_out: Driver = ctx.out_ds.GetDriver()
        drv_out.Delete(ctx.out_ds.GetDescription())
        if ctx.req.region:
            # 存在region说明裁切过，删除裁切生成的文件
            drv_in: Driver = ctx.in_ds.GetDriver()
            drv_in.Delete(ctx.in_ds.GetDescription())

    def _gdal2pillow(self, gdal_arr: np.array) -> np.array:
        """Convert GDAL array to Pillow image"""
        # gdal reads image as (band, x, y), but pillow expects (x, y, band)
        return np.rollaxis(gdal_arr, 0, 3)

    def _pillow2gdal(self, pillow_arr: np.array) -> np.array:
        """Convert Pillow image to GDAL array"""
        # pillow reads image as (x, y, band), but gdal expects (band, x, y)
        return np.rollaxis(pillow_arr, 2, 0)

    def _format_response(self, ds):
        s3_uri = ds.GetDescription()
        bucket, key = self._get_bucket_and_key_from_s3(s3_uri)
        return {
            "bucket": bucket,
            "key": key
        }

    def _get_bucket_and_key_from_s3(self, s3_uri: str):
        s3_uri = s3_uri.replace("/vsis3/", "").replace("s3://", "")  # 去掉前缀
        return s3_uri.split("/", 1)

    def _post_process_one_image(self, ctx: InferenceContext) -> np.array:
        """Postprocess inference output, override this method to customize postprocess"""
        return ctx.output_data

    def _preprocess_one_request(self, req: CommonRequest) -> InferenceContextGenerator:
        in_ds: Dataset = self._get_s3_gdal_dataset(req)
        buffer_ds: Dataset = self._create_buffer_gdal_dataset(in_ds)

        ctx = InferenceContext(in_ds, req, req.band_list, buffer_ds)
        for img_arr, x_off, y_off in ctx.yield_block_data(self.image_size, self.image_size):
            ctx.raw_data = img_arr
            ctx.preprocessed_data = self._preprocess_image(img_arr)
            ctx.x_off = x_off
            ctx.y_off = y_off
            yield self._preprocess_image_hook(ctx)

    def _preprocess_image_hook(self, ctx) -> InferenceContext:
        return ctx

    def _get_gdal_s3_uri(self, s3_uri: str) -> str:
        """Get the gdal s3 uri"""
        return s3_uri.replace("s3://", "/vsis3/")

    def _get_output_without_suffix_s3_uri_from_in_s3(self, s3_uri: str) -> str:
        """Get the output s3 uri"""
        s3_uri = s3_uri.replace("s3://", "").replace("/vsis3/", "")
        bucket, *_, name = s3_uri.split("/")
        name = name.split(".")[0]

        return "/vsis3/" + bucket + "/" + self.RESULT_OSS_PREFIX + uuid.uuid4().hex + "/" + name

    def _preprocess_image(self, image: np.array) -> Any:
        """Preprocess an image"""
        return image

    def _get_device_and_map_location(self):
        if torch.cuda.is_available() and self.properties.get("gpu_id") is not None:
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(self.properties.get("gpu_id"))
            )
        else:
            self.map_location = "cpu"
            self.device = torch.device(self.map_location)

    def _load_model(self):
        self.model = self._load_torchscript_model(self.model_pt_path)
        self.model.to(self.device)
        self.model.eval()

    def init4test(self, model_path, gpu=False):
        self.map_location = "cuda" if gpu else "cpu"
        self.device = torch.device(self.map_location)
        self.model_pt_path = model_path
        self._load_model()
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def _get_s3_gdal_dataset(self, req: CommonRequest) -> Dataset:
        """Get GDAL dataset from S3 URI"""
        s3_uri = req.uri
        s3_uri = s3_uri.replace("s3://", "/vsis3/")
        return gdal.Open(s3_uri, gdal.GA_ReadOnly)

    def _create_buffer_gdal_dataset(self, in_ds: Dataset) -> Dataset:
        """Create output GDAL dataset"""
        w, h = in_ds.RasterXSize, in_ds.RasterYSize
        f = NamedTemporaryFile("wb", delete=False)
        driver = gdal.GetDriverByName("GTIFF")
        dataset: Dataset = driver.Create(f.name, w, h, self.out_band_num)
        dataset.SetProjection(in_ds.GetProjection())
        dataset.SetGeoTransform(in_ds.GetGeoTransform())
        return dataset

    def _get_output_gdal_dataset(self, ctx: InferenceContext) -> Dataset:
        buffer_ds = ctx.out_ds
        if self.driver == "GTIFF":
            return buffer_ds
        driver = gdal.GetDriverByName(self.driver)
        out_name = self._get_output_without_suffix_s3_uri_from_in_s3(
            ctx.in_ds.GetDescription()) + get_suffix_from_driver(self.driver)
        output_ds = driver.CreateCopy(out_name, buffer_ds, 0)
        return output_ds

    def _parse_request(self, request) -> CommonRequest:
        """Parse request input"""
        data = request.get("data") or request.get("body")
        # {"uri": "s3_uri", "band_list": [1, 2, 3], "params": {...}}
        return CommonRequest(**data)
