import os
import uuid

import requests
from geo.Geoserver import Geoserver
from loguru import logger
from osgeo import osr, ogr, gdal
from osgeo.gdal import Dataset
from osgeo.ogr import DataSource

from mixins.io_mixin.vec_response_mixin import VectorResponseMixin


class PostgreSQL2WMSResponseMixin(VectorResponseMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geoserver_info = {
            "url": os.environ["GEOSERVER_URL"],
            "user": os.environ["GEOSERVER_USER"],
            "password": os.environ["GEOSERVER_PASSWORD"],
            "workspace": os.environ["GEOSERVER_WS"],
            "store": os.environ["GEOSERVER_STORE"],
        }
        self.geoserver_client = Geoserver(self.geoserver_info['url'], self.geoserver_info['user'],
                                          self.geoserver_info['password'])

    def _raster2vec(self, raster_ds: Dataset, out_ds: DataSource):
        inband = raster_ds.GetRasterBand(1)
        prj = osr.SpatialReference()
        prj.ImportFromWkt(raster_ds.GetProjection())
        layer_name = uuid.uuid4().hex
        Polygon_layer = out_ds.CreateLayer(layer_name, srs=prj, geom_type=ogr.wkbMultiPolygon)
        newField = ogr.FieldDefn('Value', ogr.OFTInteger)
        Polygon_layer.CreateField(newField)

        gdal.Polygonize(inband, None, Polygon_layer, 0)

        for feat in Polygon_layer:
            if feat.GetField('Value') == self.bg_pixel_value:
                Polygon_layer.DeleteFeature(feat.GetFID())
        self._publish(layer_name)
        logger.info(f"Publish layer {self.geoserver_info['workspace']}:{layer_name} to geoserver")
        return f"{self.geoserver_info['workspace']}:{layer_name}"

    def _publish(self, layer_name: str):
        self.geoserver_client.publish_featurestore(workspace=self.geoserver_info['workspace'],
                                                   store_name=self.geoserver_info['store'],
                                                   pg_table=layer_name)
        if self.style_name:
            self.__apply_style(f"{self.geoserver_info['workspace']}:{layer_name}")

    def __apply_style(self, layer_name: str):
        layer_info = {
            "layer": {
                "defaultStyle": {
                    "name": self.style_name
                }
            }
        }
        r = requests.put(self.geoserver_info['url'] + f'/rest/layers/{layer_name}',
                         auth=(self.geoserver_info['user'], self.geoserver_info['password']),
                         headers={'Content-type': 'application/json'},
                         json=layer_info)
        if r.status_code == 200:
            logger.info(f"Apply style {self.style_name} to {layer_name}")
        else:
            logger.warning(f"Cannot apply style {self.style_name} to {layer_name}")
