import os

import cv2
import numpy as np
import torch
from skimage import transform, color
from torch.autograd import Variable
from torchvision import transforms
from ts.torch_handler.base_handler import logger

from data_types import Arr, InferenceContext
from handlers.geojson_response_handler import GeoJsonResponseHandler
from mixins.preprocess_mixin.mask_mixin import MaskByGeoJsonMixin


class RescaleT(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')

        return img


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, image):
        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                    np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                    np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                    np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                    np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                    np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                    np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                    np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                    np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                    np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))

        return torch.from_numpy(tmpImg)


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


class RoadHandler(MaskByGeoJsonMixin, GeoJsonResponseHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms.Compose([RescaleT(256), ToTensorLab(flag=0)])

    def _preprocess_image(self, image) -> Arr:
        image = self._gdal2pillow(image)  # gdal array to pillow array
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        image = self.transforms(image)
        image = image.unsqueeze(0)
        image = image.type(torch.FloatTensor)

        if self.map_location == "cuda":
            image = Variable(image.cuda())
        else:
            image = Variable(image)
        return image

    def _post_process_one_image(self, ctx: InferenceContext) -> np.array:
        output_data = ctx.output_data
        data, *_ = output_data
        pred = data[:, 0, :, :]
        pred = normPRED(pred)
        predict = pred.squeeze()
        predict_np = predict.cpu().data.numpy()
        predict_np = predict_np * 255

        predict_np_resize = cv2.resize(predict_np, (self.image_size, self.image_size))
        predict_np_resize[predict_np_resize > 125] = 255
        predict_np_resize[predict_np_resize <= 125] = 0
        return predict_np_resize.astype(np.uint8)


# _service = RoadHandler(image_size=512, out_band_num=1)


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None

    req = _service.parse_request(data)
    try:
        data = _service.preprocess(req)
        data = _service.inference(data)
        data = _service.postprocess(data)

        name = data['s3'].split("/")[-1]

        _service.notifier_client.success(name, req.task_id, name)
        logger.info(f"Successfully processed the request: {req}")
        return [data]
    except Exception as e:
        _service.notifier_client.failure(req.task_id)
        logger.error(f"Failed to process the request: {e}, request: {req}")
        return [{"error": str(e)}]


if __name__ == '__main__':
    os.environ["MINIO_ACCESS_KEY"] = "root"
    os.environ["MINIO_SECRET_KEY"] = "admin@123"
    os.environ["MINIO_ENDPOINT"] = "localhost:9000"

    req = [{
        "body":
            {
                "task_id": 1,
                "uri": "s3://rirs-dlplat/H50F016002_Level_18.tif",
                # "uri": "s3://rirs-dlplat/data-resource/0e7cf2b1-474b-44ea-a3ab-166f5ee685d6/道路.tiff",
                "band_list": [1, 2, 3],
                "params": {...},
                "region": {
                    "type": "FeatureCollection",
                    "name": "mask",
                    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                    "features": [
                        {"type": "Feature", "properties": {"id": 1},
                         "geometry": {"type": "MultiPolygon", "coordinates": [[[[
                             114.202822097349085,
                             30.740889238756761],
                             [
                                 114.241723136030998,
                                 30.734904463574924],
                             [
                                 114.233823232790982,
                                 30.696003424893004],
                             [
                                 114.197316104181795,
                                 30.713119881913048],
                             [
                                 114.202822097349085,
                                 30.740889238756761]]]]}}
                    ]
                }
            }
    }]

    handler = RoadHandler(image_size=512, out_band_num=1)
    handler.init4test(
        model_path="/home/i/PycharmProjects/torch_handlers/examples/model_files/minty_road/minty_road.torchscript",
        gpu=True)
    req = handler.parse_request(req)
    try:
        data = handler.preprocess(req)
        data = handler.inference(data)
        data = handler.postprocess(data)
        name = data['s3'].split("/")[-1]
        # handler.notifier_client.success(name, req.task_id, name)
    except Exception as e:
        # handler.notifier_client.failure(req.task_id)
        pass
    print(data)
