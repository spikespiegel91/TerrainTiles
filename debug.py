import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import Window
from rasterio.warp import transform_bounds, calculate_default_transform, reproject
from rasterio.enums import Resampling
from matplotlib import pyplot
import mercantile

from PIL import Image
import numpy as np
import os

SOURCE_RASTER = 'demos/RGB.byte.tif'
TARGET_RASTER = 'Temp/RGB_mercator.tif'
TARGET_CRS = 'EPSG:3857'
GEODETIC_CRS = 'EPSG:4326'

        

# TODO: read and image from a path ans save it into a geoTif in 4326. The extent of the image is known in 4326
#   "CRS_AUTHID": "EPSG:4326",
#   "EXTENT": "-70.4699375340844796,18.3200383409215775 : -70.3973476670891074,18.3717854791226110",
#   "HAS_NODATA_VALUE": false,
#   "HEIGHT_IN_PIXELS": 10000,
#   "PIXEL_HEIGHT": 1e-05,
#   "PIXEL_WIDTH": 1e-05,
#   "WIDTH_IN_PIXELS": 10000,
#   "X_MAX": -70.39734766708911,
#   "X_MIN": -70.46993753408448,
#   "Y_MAX": 18.37178547912261,
#   "Y_MIN": 18.320038340921577,


#####################################################
# FIXME: revisar, lectura de un .png georeferenciado y conversion a GeoTiff con rasterio
src_path = "demos/Bani_Texture.png"
with rasterio.open(src_path) as src:
    data = src.read() # indexes=3
    name = src.name
    affine_trans = src.transform
    meta_Data = src.meta
    

    print(f"Successfully opened image file: {src_path}")

X_MAX= -70.39734766708911
X_MIN= -70.46993753408448
Y_MAX= 18.37178547912261
Y_MIN= 18.320038340921577

img_bands, img_h, img_w = data.shape

img_transform = rasterio.transform.from_bounds(
    X_MIN,Y_MIN,X_MAX,Y_MAX,img_w,img_h)

img_dtype = data.dtype

img_crs = GEODETIC_CRS

# r = data[0]
# b = data[1]
# g = data[2]

#
#  dst.write(data) is the way to write a 3D array to raster. 
#  However, rasterio is expecting data in (bands, rows, cols).
#  
#


out_path = "Temp/Bani_Texture_geo.tif"
with rasterio.open(
    out_path,
    'w',
    driver='GTiff',
    height=img_h,
    width=img_w,
    count=4,
    dtype=img_dtype,
    crs=img_crs,
    transform=img_transform,
) as dst:
    # dst.write(data, 1)
    # dst.write(r, 1)
    # dst.write(g, 2)
    # dst.write(b, 3)
    #
    dst.write(data)
    # dst.write(data, 1)
    # dst.write(data, 2)
    # dst.write(data, 3)


from rasterio.plot import show

with rasterio.open(out_path) as src:
    data = src.read()
    show(data, transform=src.transform)
    print(f"Successfully read and displayed georeferenced image: {out_path}")
