import rasterio
from rasterio.plot import show
from PIL import Image
import numpy as np

from numpy import ndarray
from typing import Literal

from .carto import parse_metadata_from_src


def elevation2terrainRGB(elevation_data: ndarray) -> Image.Image:

    if elevation_data.ndim != 2:
        raise ValueError("Input elevation data must be a 2D array")

    # Terrarium format encoding parameters:
    # elevation = (R*256 + G  + B/256) - 32768

    # offset = - 32768
    # rScaler = 256
    # gScaler = 1
    # bScaler = 1/256

    value = elevation_data.astype(np.uint32) + 32768
    r = np.floor(value / 256)
    g = np.floor(value % 256)
    b = np.round((value - np.floor(value)) * 256)

    rgb_data = np.stack([r, g, b], axis=-1).astype(np.uint8)

    return Image.fromarray(rgb_data, mode="RGB")


def terrainRGB2elevation(rgb_image: Image.Image) -> ndarray:

    rgb_data = np.array(rgb_image)
    r = rgb_data[:, :, 0].astype(np.uint32)
    g = rgb_data[:, :, 1].astype(np.uint32)
    b = rgb_data[:, :, 2].astype(np.uint32)

    return (r * 256 + g + b / 256) - 32768


def data2img(
    data: ndarray,
    encoder: None | Literal["terrain-rgb", "greyscale", "color"] = None,
    encoder_settings: dict | None = None,
) -> Image.Image:

    if data.any():
        #############################################################################
        if data.ndim == 0:
            raise ValueError("Input array must have at least one dimension")

        if data.ndim == 3 and data.shape[0] in (1, 3, 4):
            # Rasterio returns band-first arrays; move channels to the end for PIL.
            data = np.moveaxis(data, 0, -1)

        elif data.ndim not in (2, 3):
            raise ValueError("Unsupported array shape for image conversion")

        channels = 1 if data.ndim == 2 else data.shape[-1]
        #############################################################################

        if channels == 1:
            mode = "L"
        elif channels == 3:
            mode = "RGB"
        elif channels == 4:
            mode = "RGBA"
        else:
            raise ValueError("Unsupported number of channels for PIL image")

        if encoder == None:
            data = np.clip(data, 0, 255).astype(np.uint8)
            img = Image.fromarray(data, mode)

        elif encoder == "terrain-rgb":
            img = elevation2terrainRGB(data)

        elif encoder == "greyscale":
            offset = encoder_settings.get("offset", 0)
            rScaler = encoder_settings.get("rScaler", 1)

            data = np.clip((data - offset) * rScaler, 0, 255).astype(np.uint8)
            img = Image.fromarray(data, "L")

        return img


def data2GeoTif(data: ndarray, out_path: str, latlongBounds: dict, out_filename:str = None, useBuffer: bool = False) -> dict:

    img_bands, img_h, img_w = data.shape

    # X_MIN = latlongBounds.get("X_MIN", -70.46993753408448)
    # X_MAX = latlongBounds.get("X_MAX", -70.39734766708911)
    # Y_MIN = latlongBounds.get("Y_MIN", 18.320038340921577)
    # Y_MAX = latlongBounds.get("Y_MAX", 18.37178547912261)

    X_MIN = latlongBounds["X_MIN"]
    X_MAX = latlongBounds["X_MAX"]
    Y_MIN = latlongBounds["Y_MIN"]
    Y_MAX = latlongBounds["Y_MAX"]

    img_transform = rasterio.transform.from_bounds(X_MIN, Y_MIN, X_MAX, Y_MAX, img_w, img_h)

    img_dtype = data.dtype

    # img_crs = GEODETIC_CRS
    _GEODETIC_CRS = "EPSG:4326"

    if useBuffer:
        
        out_memfile = rasterio.MemoryFile(filename=out_filename)
        with out_memfile.open(
            driver="GTiff",
            height=img_h,
            width=img_w,
            count=img_bands,  # 4,
            dtype=img_dtype,
            crs=_GEODETIC_CRS,
            transform=img_transform,
        ) as dst:
            dst.write(data)
            parsed_data = parse_metadata_from_src(dst)
    else:

        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=img_h,
            width=img_w,
            count=img_bands,  # 4,
            dtype=img_dtype,
            crs=_GEODETIC_CRS,
            transform=img_transform,
        ) as dst:
            dst.write(data)
            parsed_data = parse_metadata_from_src(dst)

    return out_memfile, parsed_data
        

    # with rasterio.open(out_path) as src:
    #     data = src.read()
    #     show(data, transform=src.transform)
    #     print(f"Successfully read and displayed georeferenced image: {out_path}")

    # return metadata
