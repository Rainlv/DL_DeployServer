from osgeo import osr, ogr, gdal
from osgeo.gdal import Dataset
from osgeo.ogr import DataSource


class VectorResponseMixin:
    def _raster2vec(self, raster_ds: Dataset, out_ds: DataSource):
        inband = raster_ds.GetRasterBand(1)
        prj = osr.SpatialReference()
        prj.ImportFromWkt(raster_ds.GetProjection())
        Polygon_layer = out_ds.CreateLayer("", srs=prj, geom_type=ogr.wkbMultiPolygon)
        newField = ogr.FieldDefn('Value', ogr.OFTInteger)
        Polygon_layer.CreateField(newField)
        gdal.Polygonize(inband, None, Polygon_layer, 0)
        return out_ds
