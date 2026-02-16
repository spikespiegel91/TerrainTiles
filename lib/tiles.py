import mercantile
import rasterio
import rasterio.windows
from rasterio.windows import Window

from rasterio.windows import from_bounds, shape
from PIL import Image
import numpy as np
import os
from numpy import ndarray


def get_elevation_decoder():
    """
    Returns the elevation decoder parameters for Mapbox Terrain RGB format.
    Use these parameters with deck.gl TerrainLayer or other terrain libraries.

    Returns:
        dict: Decoder parameters with rScaler, gScaler, bScaler, and offset
    """
    return {
        "rScaler": 256 * 256 * 0.1,  # Red channel multiplier
        "gScaler": 256 * 0.1,  # Green channel multiplier
        "bScaler": 0.1,  # Blue channel multiplier
        "offset": -10000,  # Height offset in meters
    }


def generate_tiles(file_path, output_dir, zoom_levels=[8, 9, 10]):

    # tif = "Temp/new.tif"
    # out = "Temp/tiles"

    try:
        print(f"Attempting to open raster file: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Raster file not found: {file_path}")

        with rasterio.open(file_path) as src:
            print(f"Successfully opened raster file")
            print(f"Driver: {src.driver}")
            print(f"CRS: {src.crs}")
            print(f"Shape: {src.width} x {src.height}")
            print(f"Data type: {src.dtypes}")
            print(f"Number of bands: {src.count}")

            bounds = src.bounds
            print(f"Raster bounds: {bounds}")

            # Test reading a small sample first
            try:
                print("Testing data read...")
                test_data = src.read(1, window=rasterio.windows.Window(0, 0, 100, 100))
                print(f"Test read successful. Sample shape: {test_data.shape}")
                print(f"Sample data range: {np.nanmin(test_data):.2f} to {np.nanmax(test_data):.2f}")

                # Check for no-data values
                if hasattr(src, "nodata") and src.nodata is not None:
                    print(f"NoData value: {src.nodata}")
                    valid_data = test_data[test_data != src.nodata] if src.nodata is not None else test_data
                    if len(valid_data) > 0:
                        print(f"Valid data range: {np.nanmin(valid_data):.2f} to {np.nanmax(valid_data):.2f}")
                    else:
                        raise ValueError("No valid data found in test sample")

            except Exception as e:
                raise RuntimeError(f"Failed to read test data from raster: {str(e)}")

            # Get data range for information (but don't use for normalization)
            try:
                print("Reading full dataset for range analysis...")
                global_data = src.read(1)

                # Fill NoData values with 0 (sea level) before processing
                if hasattr(src, "nodata") and src.nodata is not None:
                    print(f"NoData value detected: {src.nodata}")
                    nodata_count = np.sum(global_data == src.nodata)
                    print(f"Filling {nodata_count} NoData pixels with 0m (sea level)")
                    global_data = np.where(global_data == src.nodata, 0, global_data)

                global_min = np.nanmin(global_data)
                global_max = np.nanmax(global_data)

                print(f"Elevation data range after NoData fill: {global_min:.2f}m to {global_max:.2f}m")
                print("Encoding using Mapbox Terrain RGB format (-10000m to +6553.5m range)")

            except Exception as e:
                raise RuntimeError(f"Failed to read full dataset: {str(e)}")

            for z in zoom_levels:
                print(f"Generating tiles for zoom level {z}")
                tiles = mercantile.tiles(*bounds, zooms=[z])

                tile_count = 0
                for t in tiles:
                    try:
                        bbox = mercantile.bounds(t)

                        window = from_bounds(bbox.west, bbox.south, bbox.east, bbox.north, src.transform)

                        data = src.read(
                            1, window=window, out_shape=(256, 256), resampling=rasterio.enums.Resampling.bilinear
                        )

                        # Fill NoData values with 0m (sea level) for consistent encoding
                        if hasattr(src, "nodata") and src.nodata is not None:
                            data = np.where(data == src.nodata, 0, data)

                        # print(f"Processing tile z={z}, x={t.x}, y={t.y} with data range: {np.nanmin(data):.2f}m to {np.nanmax(data):.2f}m")

                        # Encode elevation using Mapbox Terrain RGB format
                        # Formula: height = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
                        # This gives us ~1cm precision over a -10000m to +6553.5m range
                        if data.size > 0:
                            # Convert elevation to the encoded integer format
                            # height_encoded = (height + 10000) / 0.1
                            encoded_heights = np.clip((data + 10000) / 0.1, 0, 16777215).astype(np.uint32)

                            # Extract RGB components
                            r = (encoded_heights // (256 * 256)) % 256
                            g = (encoded_heights // 256) % 256
                            b = encoded_heights % 256

                            # Stack into RGB array
                            rgb_data = np.stack([r, g, b], axis=-1).astype(np.uint8)
                            img = Image.fromarray(rgb_data, mode="RGB")
                        else:
                            # Empty tile - sea level (0m) encoded
                            sea_level_encoded = int((0 + 10000) / 0.1)
                            r = (sea_level_encoded // (256 * 256)) % 256
                            g = (sea_level_encoded // 256) % 256
                            b = sea_level_encoded % 256
                            rgb_data = np.full((256, 256, 3), [r, g, b], dtype=np.uint8)
                            img = Image.fromarray(rgb_data, mode="RGB")

                        path = f"{output_dir}/{z}/{t.x}"
                        os.makedirs(path, exist_ok=True)
                        img.save(f"{path}/{t.y}.png")

                        tile_count += 1

                    except Exception as tile_error:
                        print(f"Error processing tile z={z}, x={t.x}, y={t.y}: {str(tile_error)}")
                        continue

                print(f"Generated {tile_count} tiles for zoom level {z}")

    except Exception as e:
        print(f"Error in generate_tiles: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


#####################################################
from io import BytesIO
from flask import send_file
from PIL import Image
import os


def serve_image_from_buffer(img: Image.Image):
    img_io = BytesIO()
    img.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")


def check_file_extension(filename: str, valid_extensions: list[str] = []) -> bool:
    filename_ext = filename.split(".")[-1]

    if filename_ext not in valid_extensions:
        return False
    return True


def check_path(filepath: str) -> bool:
    return os.path.exists(filepath)


def get_filepath(dirpath: str, filename: str) -> str:
    return os.path.join(dirpath, filename)


def read_tiledata_from_raster(
    filename: str,
    dirpath: str = "/Temp",
    indexes: int | list = 1,
    tile_window: list[Window] | None = None,
    out_shape: tuple | None = (256, 256),
    **kwargs,
) -> list[ndarray]:

    _valid_ext = ["tif"]

    if not check_file_extension(filename, _valid_ext):
        raise TypeError("Invalid file extension.")

    src_path = get_filepath(dirpath, filename)

    if not check_path(src_path):
        raise ValueError(" Invalid file, file does not exist")

    fill_value = kwargs.get("fill_value", 0)
    boundless = kwargs.get("boundless", True)
    # out_shape = kwargs.get("out_shape", (256, 256))

    with rasterio.open(src_path) as src:
        
        if (tile_window is None) or (out_shape is None):
            # la ventana de la tila tiene que ser cuadrada para
            # que el out_shape de 256x256 estandar de tilas funcione correctamente
            # sin distorsionar la imagen
            src_window = from_bounds(*src.bounds, transform=src.transform)
            min_size = min(src_window.width, src_window.height)

            tile_window = [src_window.crop(min_size, min_size)]
            out_shape = ( int(min_size), int(min_size) ) #shape(tile_window[0].round_shape())

        data = []
        for tile in tile_window:
            # print(f"Reading data for tile window: {tile}")
        
            tile_data = src.read(
                indexes=indexes,
                window=tile,
                out_shape=out_shape,
                boundless=boundless,  # Read data even if window is partially outside raster
                fill_value=fill_value,  # Fill empty areas with specified fill value
                # resampling=rasterio.enums.Resampling.bilinear
                resampling=rasterio.enums.Resampling.cubic,
                **kwargs,
            )
            data.append(tile_data)

    return data


def data2img(data: ndarray, img_encoder: any = None) -> Image.Image:

    # Process only if there is non-empty data
    if data.any():
        img_data = data.astype(np.uint8)
        img = Image.fromarray(img_data)

        return img


def get_raster_metadata(
    filename: str,
    dirpath: str = "/Temp",
) -> dict:
    _valid_ext = ["tif"]

    if not check_file_extension(filename, _valid_ext):
        raise TypeError("Invalid file extension.")

    src_path = get_filepath(dirpath, filename)

    if not check_path(src_path):
        raise ValueError(" Invalid file, file does not exist")

    with rasterio.open(src_path) as src:

        metadata = {
            "driver": src.driver,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "dtypes": src.dtypes,
            "count": src.count,
            "bounds": src.bounds,
            "transform": src.transform, # Affine transformation matrix for georeferencing pixels to geographic coordinates
            "nodata": src.nodata if hasattr(src, "nodata") else None,
        }

    return metadata

def Pixel2Projected(PixelCoords: list[tuple], transform):
    """
    Convert pixel coordinates (row, col) to projected coordinates (x, y) using the affine transform.

    (0,0) -------> row
      |
      |  img (pixel coordinates)                                                                         
      |                                    y   
      v col                               |       projected coordinates (x, y)
                                     (0,0)|____ x

                                     
    Args:
        PixelCoords (list of tuples): List of pixel coordinates as (row, col)
        transform (Affine): Affine transformation matrix from rasterio

    Returns:
        list of tuples: List of projected coordinates as (x, y)
    """
    transformer = rasterio.transform.AffineTransformer(transform)

    ProjectedCoord = [transformer.xy(row, col) for row, col in PixelCoords]

    return ProjectedCoord

def Projected2Pixel(ProjectedCoords: list[tuple], transform):
    """
    Convert projected coordinates (x, y) to pixel coordinates (row, col) using the affine transform.
                        
    Args:
        ProjectedCoords (list of tuples): List of projected coordinates as (x, y)
        transform (Affine): Affine transformation matrix from rasterio

    Returns:
        list of tuples: List of pixel coordinates as (row, col)
    """
    transformer = rasterio.transform.AffineTransformer(transform)

    PixelCoord = [transformer.rowcol(x, y) for x, y in ProjectedCoords]

    return PixelCoord


# TODO: metodo para trasnformar coordenadas entre sistemas de coordenadas,
# necesario para pasar de las coordenadas de los bounds de nuestro raster al sistema
# de coordendas geograficas (lat/lon) que espera mercantile para generar las tilas, 
# y luego transformar de nuevo las coordenadas de cada tila al sistema de coordenadas 
# del raster para leer los datos correctos de cada tila
#   x,y  a lon,lat  a x,y
# rasterio.warp.transform_bounds()

# metodo para transformar coordenadas simples?

# TODO metodo para reproyectar completamente el raster a otro sistema de coordenadas 
# ( el de mercantile por ejemplo) y luego generar las tilas a partir de ese raster reproyectado,
#  esto podria ser mas preciso y evitar problemas de distorsion en las tilas a menor zoom,
#  aunque podria ser mas costoso computacionalmente. 
#
# - Opcion 1: reproyectar el raster completo y guardarlo en disco
# - Opcion 2: reproyectar al completo pero guardarlo en buffer y generar las tilas a partir de ese buffer en memoria, esto podria ser mas rapido pero podria consumir mucha memoria dependiendo del tamaño del raster
#  en lugar de leer de disco con read_tiledata_from_raster() -
#  --> leemos desde el buffer read_tiledata_from_rasterBuffer()

# TODO: metodo para coordinar la generacion de tilas, aseguranto que estan en el sistema
# de coordenadas correcto, y que se guardan en la estructura de carpetas correcta (z/x/y.png)
# ------> Class