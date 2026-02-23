import mercantile
import rasterio
from rasterio.windows import Window
from rasterio.windows import from_bounds
from PIL import Image
import os
from numpy import ndarray
from typing import Literal

from .io import *
from .carto import (
    bounds2Geographic,
    boundsFromGeographic,
    get_raster_metadata,
    get_Geographic_window_from_bounds,
    reproject_raster2Geographic,
)
from .encoders import data2img, data2GeoTif


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


def get_tiles(bounds, bounds_crs, zoom_levels=[1, 2]):
    # GEODETIC_CRS = "EPSG:4326"
    # geodetic_bounds = transform_bounds(bounds_crs, GEODETIC_CRS, bounds)
    geo_bounds = bounds2Geographic(bounds_crs, bounds)
    # mercantile_bounds = bounds2Mercator(bounds_crs, bounds)
    # mercator_crs_ = "EPSG:3857"

    tiles = []
    for z in zoom_levels:
        tiles_z = mercantile.tiles(geo_bounds[0], geo_bounds[1], geo_bounds[2], geo_bounds[3], zooms=[z])

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
    def __init__(
        self, raster_dir: str, filename: str, output_dir: str, mode: Literal["tif", "png"] = "tif", settings: dict = {}
    ):
        """_summary_

        Args:
           settings (dict, optional): _description_. Defaults to {}.
            {x_MIN, x_MAX, y_MIN, y_MAX} in geographic coordinates for raster data extraction from texture png mode

        """
        self.raster_dir = raster_dir
        self.filename = filename
        self.mode = mode

        self.output_dir = output_dir
        self.tiles_name = "tiles"
        self.tiles_data = None
        # self.raster_path = get_filepath(raster_dir, raster_name)
        # self.Geo_4326_reprojected_tag = 'geo_4326'
        # self.Mercator_3857_reprojected_tag = 'mercator_3857'
        self.useBuffer = False
        # self._tile_shape = (256, 256)
        self._tile_shape = (512, 512)
        self.zoom_levels = [1, 2, 3, 4]
        self.settings = settings

        self._check_mode()
        self._update_tiles_path()

        # si es png, reload_aster_data tiene que transformar primero de png a tif
        if self.mode == "png":
            self.texture_name = filename  # .png
            self.raster_name = change_file_extension(self.texture_name, "tif")

        elif self.mode == "tif":
            self.texture_name = None
            self.raster_name = filename  # .tif

        self.reload_raster_data()

    def set_zoom(self, zoom_levels: list[int]):
        self.zoom_levels = zoom_levels

    def set_tiles_name(self, name: str):
        self.tiles_name = name
        self._update_tiles_path()

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

    
    def generate_elevation_tiles(self):
        self.save_tiles_png(indexes=1, encoder="terrain-rgb")
        return
    
    def generate_texture_tiles(self):
        self.save_tiles_png(indexes=None, encoder=None)
        return
    
    def save_tiles_png(
        self,
        indexes=1, # 1, None, [1,2,3]
        encoder: None | Literal["terrain-rgb", "greyscale"] = None,
        encoder_settings: dict | None = None,
    ):
        """_summary_

        Args:
            None (_type_): _description_
            indexes (int, optional): _description_. Defaults to 1.
            encoder_settings (dict | None, optional): _description_. Defaults to None.

            For elevation data: 
            indexes = 1
            encoder = "terrain-rgb"

            For texture layer:
            indexes = [1,2,3] or None (all bands)
            encoder = None


        """
        
        self._clean_tiles_output()

        if self.tiles_data is None:
            self._get_tiles_data()

        all_windows = [t["window"] for t in self.tiles_data]

        max_batch_size = 100
        num_windows = len(all_windows)
        batch_size = min(max_batch_size, num_windows)

        for i in range(0, num_windows, batch_size):
            batch_windows = all_windows[i : i + batch_size]

            sliced_rasterdata = read_tiledata_from_raster(
                self.out_raster_name,
                self.output_dir,
                indexes,
                tile_window=batch_windows,
                out_shape=self._tile_shape,
            )

            img_all = [
                (index, self._encode_data(slicedata, encoder, encoder_settings))
                for index, slicedata in enumerate(sliced_rasterdata)
            ]

            for index, img in img_all:

                if img is None:
                    continue

                t = self.tiles_data[index + i]
                z = t["z"]
                x = t["x"]
                y = t["y"]

                path = f"{self.tiles_path}/{z}/{x}"
                os.makedirs(path, exist_ok=True)
                img.save(f"{path}/{y}.png")

    def reload_raster_data(self):
        self._reset_tiles_data()
        try:

            if self.mode == "png":
                self._raster_from_texture_4326()

            elif self.mode == "tif":
                self.src_raster_metada = get_raster_metadata(self.raster_name, self.raster_dir)
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

    def _raster_from_texture_4326(self):
        latlongBounds = {
            "X_MIN": self.settings.get("X_MIN", None),
            "X_MAX": self.settings.get("X_MAX", None),
            "Y_MIN": self.settings.get("Y_MIN", None),
            "Y_MAX": self.settings.get("Y_MAX", None),
        }

        if None in latlongBounds.values():
            raise ValueError("Missing geographic bounds in settings for texture png mode")

        imgData = read_image_from_path(self.texture_name, self.raster_dir)

        if self.useBuffer:
            # TODO
            # Implement logic to reproject raster to geographic coordinates and store in buffer
            pass
        else:
            self.out_raster_name = self.raster_name
            self.out_raster_path = os.path.join(self.output_dir, self.raster_name)

            self.out_raster_metadata = data2GeoTif(imgData, self.out_raster_path, latlongBounds)
            self.BufferData = None

        return

    def _reproject_lnglat_4326(self):

        if self.useBuffer:
            # TODO
            # Implement logic to reproject raster to geographic coordinates and store in buffer
            pass
        else:
            self.out_raster_path, self.out_raster_name = reproject_raster2Geographic(
                self.raster_name, self.raster_dir, self.output_dir
            )
            self.out_raster_metadata = get_raster_metadata(self.out_raster_name, self.output_dir)
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
            tile_window = get_Geographic_window_from_bounds(bounds_crs, bounds, affine_coord2pixel)

            tiles_data[index].update({"window": tile_window})

        self.tiles_data = tiles_data

    def _encode_data(
        self,
        data: ndarray,
        encoder: None | Literal["terrain-rgb", "greyscale", "color"] = None,
        settings: dict | None = None,
    ) -> Image.Image:

        return data2img(data, encoder, settings)

    def _update_tiles_path(self):
        self.tiles_path = os.path.join(self.output_dir, self.tiles_name)

    def _reset_tiles_data(self):
        self.tiles_data = None

    def _clean_tiles_output(self):
        # tiles_path = f"{self.output_dir}/tiles"

        if os.path.exists(self.tiles_path):
            for root, dirs, files in os.walk(self.tiles_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            print(f"Cleaned existing tiles in {self.tiles_path}")
        else:
            print(f"No existing tiles to clean in {self.tiles_path}")


    def _check_mode(self):
        validExtension = check_file_extension(self.filename, self.mode)
        if not validExtension:
            raise ValueError(f"Invalid mode: {self.mode} for file: {self.filename}")

