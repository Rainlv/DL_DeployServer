import json
import uuid
from tempfile import NamedTemporaryFile

from osgeo import gdal, ogr
from osgeo.gdal import Dataset

from data_types import CommonRequest


class MaskByGeoJsonMixin:
    def _get_s3_gdal_dataset(self, req: CommonRequest) -> Dataset:
        raw_ds: Dataset = super()._get_s3_gdal_dataset(req)
        if not req.region:
            return raw_ds
        mask_geojson_ds = ogr.Open(json.dumps(req.region))
        return self._mask_image_by_geojson(mask_geojson_ds, raw_ds)

    def _mask_image_by_geojson(self, mask_ds, raw_raster_ds: Dataset) -> Dataset:
        """Mask an image by geojson"""
        mem_path = f"/vsimem/{str(uuid.uuid4())}.shp"
        drv = ogr.GetDriverByName('ESRI Shapefile')
        drv.CopyDataSource(mask_ds, mem_path)
        f = NamedTemporaryFile('w+t', encoding='utf-8', errors='ignore', suffix='.tif', delete=False)
        return gdal.Warp(destNameOrDestDS=f.name,
                         srcDSOrSrcDSTab=raw_raster_ds,
                         cutlineDSName=mem_path, format="GTiff", cropToCutline=True)
        # return ds


if __name__ == '__main__':
    geojson = {
        "type": "FeatureCollection",
        "name": "mask",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {"type": "Feature", "properties": {"id": 1}, "geometry": {"type": "MultiPolygon", "coordinates": [[[[
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

    req = CommonRequest(uri="/home/i/Downloads/test_data/H50F016002_Level_18.tif", band_list=[], region=geojson)

    ds = MaskByGeoJsonMixin()._get_s3_gdal_dataset(req)
    print(ds)
    # mask_ds = ogr.Open(geojson)
    # drv = ogr.GetDriverByName('ESRI Shapefile')
    # drv.CopyDataSource(mask_ds, "/vsimem/mask.shp")
    # gdal.Warp(destNameOrDestDS="/home/i/PycharmProjects/torch_handlers/out/mask.tif",
    #           srcDSOrSrcDSTab="/home/i/Downloads/test_data/H50F016002_Level_18.tif",
    #           cutlineDSName="/vsimem/mask.shp", format="GTiff", cropToCutline=True)
