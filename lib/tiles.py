import mercantile
import rasterio
import rasterio.windows
from rasterio.windows import Window

from rasterio.windows import from_bounds, shape
from PIL import Image
import numpy as np
import os
from numpy import ndarray
from typing import Literal

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
                print(
                    f"Sample data range: {np.nanmin(test_data):.2f} to {np.nanmax(test_data):.2f}"
                )

                # Check for no-data values
                if hasattr(src, "nodata") and src.nodata is not None:
                    print(f"NoData value: {src.nodata}")
                    valid_data = (
                        test_data[test_data != src.nodata]
                        if src.nodata is not None
                        else test_data
                    )
                    if len(valid_data) > 0:
                        print(
                            f"Valid data range: {np.nanmin(valid_data):.2f} to {np.nanmax(valid_data):.2f}"
                        )
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

                print(
                    f"Elevation data range after NoData fill: {global_min:.2f}m to {global_max:.2f}m"
                )
                print(
                    "Encoding using Mapbox Terrain RGB format (-10000m to +6553.5m range)"
                )

            except Exception as e:
                raise RuntimeError(f"Failed to read full dataset: {str(e)}")

            for z in zoom_levels:
                print(f"Generating tiles for zoom level {z}")
                tiles = mercantile.tiles(*bounds, zooms=[z])

                tile_count = 0
                for t in tiles:
                    try:
                        bbox = mercantile.bounds(t)

                        window = from_bounds(
                            bbox.west, bbox.south, bbox.east, bbox.north, src.transform
                        )

                        data = src.read(
                            1,
                            window=window,
                            out_shape=(256, 256),
                            resampling=rasterio.enums.Resampling.bilinear,
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
                            encoded_heights = np.clip(
                                (data + 10000) / 0.1, 0, 16777215
                            ).astype(np.uint32)

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
                        print(
                            f"Error processing tile z={z}, x={t.x}, y={t.y}: {str(tile_error)}"
                        )
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
    filepath: str | None = None,
    **kwargs,
) -> list[ndarray]:

    _valid_ext = ["tif"]

    if filepath:
        src_path = filepath
        print(f"Reading from provided filepath: {filepath}")
        filename = os.path.basename(filepath)
        dirpath = os.path.dirname(filepath)
    else:
        src_path = get_filepath(dirpath, filename)
        print(f"Constructed filepath from dirpath and filename: {src_path}")

    if not check_file_extension(filename, _valid_ext):
        raise TypeError("Invalid file extension.")

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
            out_shape = (
                int(min_size),
                int(min_size),
            )  # shape(tile_window[0].round_shape())

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


def data2img(
    data: ndarray, encoder: None | Literal["terrain-rgb", "greyscale", "color"] = None
) -> Image.Image:
    # FIXME: se asume aqui que data solo tiene 1 canal? implementar encoders
    # Process only if there is non-empty data
    if data.any():
        img_data = data.astype(np.uint8)
        img = Image.fromarray(img_data)

        return img


def get_raster_metadata(
    filename: str,
    dirpath: str = "/Temp",
    filepath: str | None = None,
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
            "transform": src.transform,  # Affine transformation matrix for georeferencing pixels to geographic coordinates
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
from rasterio.warp import transform_bounds as rio_transform_bounds


def transform_bounds(src_crs, dst_crs, bounds):
    return rio_transform_bounds(src_crs, dst_crs, *bounds)


def bounds2Mercator(src_crs, bounds):
    return rio_transform_bounds(src_crs, "EPSG:3857", *bounds)


def bounds2Geographic(src_crs, bounds):
    return rio_transform_bounds(src_crs, "EPSG:4326", *bounds)


def boundsFromMercator(dst_crs, bounds):
    return rio_transform_bounds("EPSG:3857", dst_crs, *bounds)


def boundsFromGeographic(dst_crs, bounds):
    return rio_transform_bounds("EPSG:4326", dst_crs, *bounds)


def get_window_from_bounds(bounds, transform):
    return from_bounds(*bounds, transform=transform)


def get_Mercator_window_from_bounds(src_crs, src_bounds, mercator_transform):
    mercator_bounds = bounds2Mercator(src_crs, src_bounds)
    return from_bounds(*mercator_bounds, mercator_transform)


def get_Geographic_window_from_bounds(src_crs, src_bounds, geographic_transform):
    geographic_bounds = bounds2Geographic(src_crs, src_bounds)
    return from_bounds(*geographic_bounds, geographic_transform)


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


from rasterio.warp import calculate_default_transform, reproject, Resampling


def add_prefix_to_filename(filename: str, prefix: str) -> str:
    filename_ext = filename.split(".")[-1]
    filename_base = ".".join(filename.split(".")[:-1])
    return f"{filename_base}_{prefix}.{filename_ext}"


def reproject_raster(
    filename: str,
    dirpath: str = "/Temp",
    output_path: str = "/Temp",
    dst_crs: str = "EPSG:3857",
    out_prefix: str | None = None,
) -> str:

    _valid_ext = ["tif"]

    if not check_file_extension(filename, _valid_ext):
        raise TypeError("Invalid file extension.")

    src_path = get_filepath(dirpath, filename)

    if not check_path(src_path):
        raise ValueError(" Invalid file, file does not exist")

    if not check_path(output_path):
        raise ValueError(" Invalid output path, path does not exist")

    if out_prefix is None:
        out_prefix = dst_crs.split(":")[-1]

    out_filename = add_prefix_to_filename(filename, out_prefix)
    out_path = get_filepath(output_path, out_filename)

    with rasterio.open(src_path) as src:

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )
        # FIXME: este raster reproyectado se podria guardar en un buffer en memoria en lugar de en disco,
        # para luego generar las tilas a partir de ese buffer, esto podria ser mas rapido pero podria consumir mucha memoria dependiendo del tamaño del raster
        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.enums.Resampling.cubic,
                )

    return out_path, out_filename


def reproject_raster2Mercator(
    filename: str,
    dirpath: str = "/Temp",
    output_path: str = "/Temp",
):
    return reproject_raster(
        filename, dirpath, output_path, dst_crs="EPSG:3857", out_prefix="mercator_3857"
    )


def reproject_raster2Geographic(
    filename: str,
    dirpath: str = "/Temp",
    output_path: str = "/Temp",
):
    return reproject_raster(
        filename, dirpath, output_path, dst_crs="EPSG:4326", out_prefix="geo_4326"
    )


# TODO: metodo para coordinar la generacion de tilas, aseguranto que estan en el sistema
# de coordenadas correcto, y que se guardan en la estructura de carpetas correcta (z/x/y.png)
# ------> Class
# opcion 2: metodo para servir dinamicante la tila pedida, sin guardar en disco.


def get_tiles(bounds, bounds_crs, zoom_levels=[1, 2]):
    # GEODETIC_CRS = "EPSG:4326"
    # geodetic_bounds = transform_bounds(bounds_crs, GEODETIC_CRS, bounds)
    geo_bounds = bounds2Geographic(bounds_crs, bounds)
    # mercantile_bounds = bounds2Mercator(bounds_crs, bounds)
    # mercator_crs_ = "EPSG:3857"

    tiles = []
    for z in zoom_levels:
        tiles_z = mercantile.tiles(
            geo_bounds[0], geo_bounds[1], geo_bounds[2], geo_bounds[3], zooms=[z]
        )

        tiles.extend(tiles_z)

    tile_data = []
    for t in tiles:

        "return the web mercator xy box of a tile"
        xy_tile_bounds = mercantile.xy_bounds(t)

        "Returns the bounding lngLat box of a tile"
        LngLat_tile_bounds = mercantile.bounds(t)

        native_bounds = boundsFromGeographic(bounds_crs, LngLat_tile_bounds)

        tile_data.append(
            {
                "x": t.x,
                "y": t.y,
                "z": t.z,
                "mercator_bounds": xy_tile_bounds,
                "geographic_bounds": LngLat_tile_bounds,
                "native_bounds": native_bounds,
            }
        )

    return tile_data



class TileGenerator:
    def __init__(self, raster_dir: str, raster_name: str, output_dir: str):
        self.raster_dir = raster_dir
        self.raster_name = raster_name
        self.output_dir = output_dir
        # self.raster_path = get_filepath(raster_dir, raster_name)
        # self.Geo_4326_reprojected_tag = 'geo_4326'
        # self.Mercator_3857_reprojected_tag = 'mercator_3857'
        self.useBuffer = False
        self._tile_shape = (256, 256)
        self.zoom_levels = [1, 2, 3, 4]

        self.reload_raster_data()

    def set_zoom(self, zoom_levels: list[int]):
        self.zoom_levels = zoom_levels

    def get_src_metadata(self):
        return self.src_raster_metada

    def get_out_metadata(self):
        return self.out_raster_metadata

    def get_tiles_data(self):
        if self.tiles_data is None:
            self._get_tiles_data()
        return self.tiles_data

    def get_tiles_count(self):
        if self.tiles_data is None:
            self._get_tiles_data()
        return len(self.tiles_data)


    def save_tiles_png(
        self,
        indexes=1,
        encoder: None | Literal["terrain-rgb", "greyscale", "color"] = None,
    ):
        self._clean_tiles_output()

        if self.tiles_data is None:
            self._get_tiles_data()

        all_windows = [t["window"] for t in self.tiles_data]

        sliced_rasterdata = read_tiledata_from_raster(
            self.out_raster_name,
            self.output_dir,
            indexes,
            tile_window=all_windows,
            out_shape=self._tile_shape,
        )

        img_all = [
            (index, self._encode_data(slicedata, encoder))
            for index, slicedata in enumerate(sliced_rasterdata)
        ]

        for index, img in img_all:

            if img is None:
                continue

            t = self.tiles_data[index]
            z = t["z"]
            x = t["x"]
            y = t["y"]

            path = f"{self.output_dir}/tiles/{z}/{x}"
            os.makedirs(path, exist_ok=True)
            img.save(f"{path}/{y}.png")

    def reload_raster_data(self):
        self._reset_tiles_data()
        try:
            self.src_raster_metada = get_raster_metadata(
                self.raster_name, self.raster_dir
            )
            # print(f"Raster metadata: {self.src_raster_metada}")
            self._reproject_lnglat_4326()

        except Exception as e:
            raise RuntimeError(f"Failed to read raster metadata: {str(e)}")

    def _get_bounds_from_metadata(self, metadata: dict | None = None):

        try:
            _bounds = metadata["bounds"]
            _bounds_crs = metadata["crs"]
            _affine_coord2pixel = metadata["transform"]

            return _bounds, _bounds_crs, _affine_coord2pixel

        except Exception as e:
            raise RuntimeError(f"Failed to extract bounds from metadata: {str(e)}")

    def _reproject_lnglat_4326(self):

        if self.useBuffer:
            # TODO
            # Implement logic to reproject raster to geographic coordinates and store in buffer
            pass
        else:
            self.out_raster_path, self.out_raster_name = reproject_raster2Geographic(
                self.raster_name, self.raster_dir, self.output_dir
            )
            self.out_raster_metadata = get_raster_metadata(
                self.out_raster_name, self.output_dir
            )
            self.BufferData = None

    def _get_tiles_data(self):

        search_bounds, bounds_crs, affine_coord2pixel = self._get_bounds_from_metadata(self.out_raster_metadata)
        tiles_data = get_tiles(search_bounds, bounds_crs, self.zoom_levels)

        for index, data in enumerate(tiles_data):
            tile_bounds = data["geographic_bounds"]
            bounds = (
                tile_bounds.west,
                tile_bounds.south,
                tile_bounds.east,
                tile_bounds.north,
            )
            tile_window = get_Geographic_window_from_bounds(
                bounds_crs, bounds, affine_coord2pixel
            )

            tiles_data[index].update({"window": tile_window})

        self.tiles_data = tiles_data

    def _encode_data(
        self,
        data: ndarray,
        encoder: None | Literal["terrain-rgb", "greyscale", "color"] = None,
    ) -> Image.Image:

        if encoder is None:
            return data2img(data)
        elif encoder == "terrain_rgb":
            return data2img(
                data, "terrain_rgb"
            )  # TODO implementar la codificacion a formato terrain rgb
        elif encoder == "greyscale":
            return data2img(
                data, "greyscale"
            )  # TODO implementar la codificacion a formato greyscale
        elif encoder == "color":
            return data2img(
                data, "color"
            )  # TODO implementar la codificacion a formato color

        pass

    def _reset_tiles_data(self):
        self.tiles_data = None

    def _clean_tiles_output(self):
        tiles_path = f"{self.output_dir}/tiles"
        if os.path.exists(tiles_path):
            for root, dirs, files in os.walk(tiles_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            print(f"Cleaned existing tiles in {tiles_path}")
        else:
            print(f"No existing tiles to clean in {tiles_path}")

