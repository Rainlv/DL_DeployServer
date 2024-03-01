import math
from dataclasses import dataclass
from typing import TypeVar, Generator, Union

import numpy as np
import torch
from osgeo.gdal import Dataset
from osgeo.ogr import DataSource

Arr = TypeVar('Arr', np.array, torch.Tensor)


@dataclass
class CommonRequest:
    uri: str
    band_list: list
    task_id: int
    params: dict = None
    region: dict = None


class InferenceContext(object):
    def __init__(self,
                 in_ds: Dataset,
                 req: CommonRequest,
                 band_list=None,
                 out_ds: Union[Dataset, DataSource] = None,
                 raw_data: np.array = None,
                 preprocessed_data: Arr = None,
                 output_data: Arr = None,
                 x_off: int = 0,
                 y_off: int = 0):
        if band_list is None:
            band_list = [1, 2, 3]
        self.in_ds = in_ds
        self.req = req
        self.band_list = band_list
        self.out_ds = out_ds
        self.raw_data = raw_data
        self.preprocessed_data = preprocessed_data
        self.output_data = output_data
        self.x_off = x_off
        self.y_off = y_off

    def yield_block_data(self, x_size, y_size):
        """Yield data block from input dataset"""
        x_off = 0
        y_off = 0

        if self.in_ds.RasterYSize < y_size:
            y_size = self.in_ds.RasterYSize
        if self.in_ds.RasterXSize < x_size:
            x_size = self.in_ds.RasterXSize
        total_block_count = math.ceil(self.in_ds.RasterXSize / x_size) * math.ceil(self.in_ds.RasterYSize / y_size)
        block_count = 0
        while y_off < self.in_ds.RasterYSize:
            y_off = min(y_off, self.in_ds.RasterYSize - y_size)
            while x_off < self.in_ds.RasterXSize:
                x_off = min(x_off, self.in_ds.RasterXSize - x_size)
                yield self.get_block_data(x_off, y_off, x_size, y_size), x_off, y_off
                x_off += x_size
                block_count += 1
                print(f"Processing block {block_count}/{total_block_count}")
            y_off += y_size
            x_off = 0

    def get_block_data(self, xoff, yoff, xsize, ysize):
        """Get data from input dataset"""
        gdal_arr = self.in_ds.ReadAsArray(xoff, yoff, xsize, ysize, band_list=self.band_list)
        return gdal_arr


InferenceContextGenerator = TypeVar('InferenceContextGenerator', Generator[InferenceContext, None, None], None)
