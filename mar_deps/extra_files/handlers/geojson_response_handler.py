import uuid

from handlers.s3_io_handler import CommonS3Handler, InferenceContext
from mixins.io_mixin.geojson_response_mixin import GeoJsonResponseMixin


class GeoJsonResponseHandler(CommonS3Handler, GeoJsonResponseMixin):
    RESULT_OSS_PREFIX = "geodata/result"

    def _get_output_gdal_dataset(self, ctx: InferenceContext):
        s3_uri = f"/vsis3/{self.RESULT_OSS_PREFIX}/{uuid.uuid4().hex}.geojson"
        return self._raster2geojson(ctx.out_ds, s3_uri)

    def _format_response(self, ds):
        return {
            "type": "geojson",
            "data": None,
            "s3": ds.GetName()
        }
