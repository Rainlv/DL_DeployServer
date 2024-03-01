import json
import os

from osgeo import gdal, ogr, osr
from osgeo.gdal import Dataset
from ts.torch_handler.base_handler import logger


class GeoJsonResponseMixin:
    def _raster2geojson(self, raster_ds: Dataset, out_uri: str):
        inband = raster_ds.GetRasterBand(1)
        drv = ogr.GetDriverByName('GeoJson')
        Polygon = drv.CreateDataSource(out_uri)
        prj = osr.SpatialReference()
        prj.ImportFromWkt(raster_ds.GetProjection())
        Polygon_layer = Polygon.CreateLayer("", srs=prj, geom_type=ogr.wkbMultiPolygon)
        newField = ogr.FieldDefn('Value', ogr.OFTInteger)
        Polygon_layer.CreateField(newField)
        gdal.Polygonize(inband, None, Polygon_layer, 0)
        return Polygon

    def _geojson_ds2text(self, geojson_ds):
        file_path = geojson_ds.GetName()
        geojson_ds.Destroy()  # 释放文件，保证读取的数据已经完整写入文件
        with open(file_path, "r") as f:
            json_data = json.load(f)
        try:
            os.remove(file_path)
        except:
            logger.warn(f"删除临时geojson文件：{file_path}失败！")
        return json_data


if __name__ == "__main__":
    c = GeoJsonResponseMixin()
    ds = gdal.Open("/home/i/Proj/torch_serve/data/珠海_Level_18_12_1.tif")
    print(c.get_geojson_text(ds))
