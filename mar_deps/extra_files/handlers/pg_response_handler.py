import os

from osgeo import ogr, gdal

from data_types import InferenceContext
from handlers.s3_io_handler import CommonS3Handler
from mixins.io_mixin.pg_response_mixin import PostgreSQL2WMSResponseMixin


class PosrgreSQLResponseHandler(CommonS3Handler, PostgreSQL2WMSResponseMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        PostgreSQL2WMSResponseMixin.__init__(self, )
        PG_DB = os.environ["PG_DB"]
        PG_USER = os.environ["PG_USER"]
        PG_PASSWORD = os.environ["PG_PASSWORD"]
        PG_HOST = os.environ["PG_HOST"]
        PG_PORT = os.environ["PG_PORT"]
        self.PG_URI = f"PG:dbname={PG_DB} user={PG_USER} password={PG_PASSWORD} host={PG_HOST} port={PG_PORT}"

    def _get_output_gdal_dataset(self, ctx: InferenceContext):
        out_vector = self._build_vector_dataset()
        return self._raster2vec(ctx.out_ds, out_vector)

    def _build_vector_dataset(self):
        return ogr.Open(self.PG_URI)

    def _format_response(self, wms_layer_name: str):
        return {
            "type": "wms",
            "data": wms_layer_name,
        }


if __name__ == '__main__':
    ctx = InferenceContext(None, None)
    ctx.out_ds = gdal.Open("/home/i/Downloads/test_data/road_samples/label/0aY4l.png")
    handler = PosrgreSQLResponseHandler()
    handler._get_output_gdal_dataset(ctx)
