import uuid

from osgeo import ogr

from handlers.s3_io_handler import CommonS3Handler, InferenceContext
from mixins.io_mixin.geojson_response_mixin import GeoJsonResponseMixin


class GeoJsonResponseHandler(CommonS3Handler, GeoJsonResponseMixin):
    RESULT_OSS_PREFIX = "geodata/result/geojson"

    def _get_output_gdal_dataset(self, ctx: InferenceContext):
        out_vector = self._build_vector_dataset()
        return self._raster2vec(ctx.out_ds, out_vector)

    def _build_vector_dataset(self):
        out_uri = f"/vsis3/{self.RESULT_OSS_PREFIX}/{uuid.uuid4().hex}.geojson"
        drv = ogr.GetDriverByName('GeoJson')
        return drv.CreateDataSource(out_uri)

    def _format_response(self, ds):
        return {
            "type": "geojson",
            "data": ds.GetName(),
        }
