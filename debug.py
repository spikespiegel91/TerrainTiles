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


def ensure_target_crs_raster(src_path, dst_path, dst_crs):
    """Reproject src_path into dst_crs once so downstream steps share a CRS."""
    if os.path.exists(dst_path):
        return dst_path

    output_dir = os.path.dirname(dst_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=rasterio.band(dst, b),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )

    return dst_path


src = rasterio.open(SOURCE_RASTER)
array = src.read(1)
pyplot.imshow(array, cmap='pink')
pyplot.show()

src.close()


target_raster_path = ensure_target_crs_raster(SOURCE_RASTER, TARGET_RASTER, TARGET_CRS)

with rasterio.open(target_raster_path) as src:
    bounds = src.bounds
    transform = src.transform
    epsg = src.crs.to_epsg() if src.crs else None
    print(f"Raster bounds: {bounds}")
    print(f"Raster transform: {transform}")
    print(f"Raster EPSG: {epsg}")

#trasnform bounds from the raster's CRS to latlong

# print(f"Raster bounds in lat/lon: {bounds_latlon}")
xmin, ymin, xmax, ymax = transform_bounds(epsg, 4326, *bounds)
print(f"Raster bounds in lat/lon: {xmin}, {ymin}, {xmax}, {ymax}")
latlong_bounds = (xmin, ymin, xmax, ymax)

tiles8 = list(mercantile.tiles(*latlong_bounds, zooms=[8]))
print(f"Generated {len(tiles8)} tiles at zoom level 8")

print("Sample tile:", tiles8[0])
print("Sample tile:", tiles8[2])

tiles5 = list(mercantile.tiles(*latlong_bounds, zooms=[5]))
print(f"Generated {len(tiles5)} tiles at zoom level 5")

sample_tile5 = tiles8[2]
bbox = mercantile.bounds(sample_tile5)
print("Sample tile at zoom level 5:", sample_tile5)
print(f"Tile bounds: {bbox}")


# mercantile bounds are in lat/lon, so we need to transform them back to the raster's CRS for reading
tile_bounds_crs = transform_bounds(4326, epsg, bbox.west, bbox.south, bbox.east, bbox.north)
print(f"Tile bounds in raster CRS: {tile_bounds_crs}")

window_ = from_bounds(
    tile_bounds_crs[0], tile_bounds_crs[1],
    tile_bounds_crs[2], tile_bounds_crs[3],
    transform #src.transform
)

window_custom = Window(0, 0, 512, 256)

print(f"Window for tile {sample_tile5}: {window_}")
print(f"Custom window: {window_custom}")

with rasterio.open(target_raster_path) as src:
    data = src.read(
        1,
        window=window_, #window_custom,
        out_shape=(256, 256),
        resampling=rasterio.enums.Resampling.bilinear
    )

    # Visualize the tile data and the original raster for comparison
    # pyplot.imshow(array, cmap='pink')
    pyplot.imshow(data, cmap='pink')  # Overlay the tile data with some transparency
    pyplot.show()

###################################################################################
output_dir = "Temp/tiles"
zoom_levels = [5, 6, 7, 8,9,10,11,12]

# [X]FIXME: parece que algunas tilas se generan distorsionadas (las tilas de los bordes a menor zoom)
# tambien hay un efecto de desfase o diente de sierra entre tilas adyacentes a menor zoom, 
# probablemente por la forma en que se calculan los bounds y el window para cada tile?
# Solucionado cambiando de sistema de coordenadas el tiff original (copia como RGB_mercador.tif), la distorsion se debia
# a error numerico introducido al reproyectar cada tile individualmente, ahora al reproyectar el tiff completo primero se evita ese error acumulativo

with rasterio.open(target_raster_path) as src:
    # Reproject the raster's bounds to geographic coordinates for mercantile
    geodetic_bounds = transform_bounds(src.crs, GEODETIC_CRS, *src.bounds)
    print(f"Raster bounds in lat/lon (EPSG:4326): {geodetic_bounds}")

    for z in zoom_levels:
        print(f"Generating tiles for zoom level {z}")
        # Get tiles that cover the geographic bounds (mercantile expects lon/lat)
        tiles = mercantile.tiles(
            geodetic_bounds[0], geodetic_bounds[1],
            geodetic_bounds[2], geodetic_bounds[3],
            zooms=[z]
        )
        tile_count = 0

        for t in tiles:
            # Get the precise Web Mercator bounds for the tile
            mercator_tile_bounds = mercantile.xy_bounds(t)

            # Transform these precise bounds back to the raster's native CRS
            tile_bounds_native_crs = transform_bounds(
                TARGET_CRS, src.crs,
                mercator_tile_bounds.left, mercator_tile_bounds.bottom,
                mercator_tile_bounds.right, mercator_tile_bounds.top
            )

            # Create a window from the transformed bounds
            tile_window = from_bounds(
                *tile_bounds_native_crs,
                transform=src.transform
            )

            # Read the data for the window
            data = src.read(
                1,
                window=tile_window,
                out_shape=(256, 256),
                boundless=True,  # Read data even if window is partially outside raster
                fill_value=0,    # Fill empty areas with 0 (black)
                #resampling=rasterio.enums.Resampling.bilinear
                resampling = rasterio.enums.Resampling.cubic
            )

            if data.any():  # Process only if there is non-empty data
                # Normalize data to 0-255 for PNG
                if data.max() > 0:
                    # img_data = (data / data.max() * 255).astype(np.uint8)
                    img_data = data.astype(np.uint8)
                else:
                    img_data = data.astype(np.uint8)

                img = Image.fromarray(img_data)
                path = f"{output_dir}/{z}/{t.x}"
                os.makedirs(path, exist_ok=True)
                img.save(f"{path}/{t.y}.png")
                
                tile_count += 1

        print(f"Generated {tile_count} tiles at zoom level {z}")
                          
