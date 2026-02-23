import rasterio
from PIL import Image
import os
from numpy import ndarray
from io import BytesIO
from flask import send_file


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

def change_file_extension(filename: str, new_extension: str) -> str:
    filename_base = ".".join(filename.split(".")[:-1])
    return f"{filename_base}.{new_extension}"


def check_path(filepath: str) -> bool:
    return os.path.exists(filepath)


def get_filepath(dirpath: str, filename: str) -> str:
    return os.path.join(dirpath, filename)


def read_image_from_path(
    filename: str,
    dirpath: str = "/Temp",
    filepath: str | None = None, # optional
    **kwargs,
        ) -> ndarray:

    _valid_ext = ["png"]
 
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
    
    #####################################################
    with rasterio.open(src_path) as src:
        print(f"Successfully opened image file: {src_path}")
        data = src.read(**kwargs)

    return data


def add_prefix_to_filename(filename: str, prefix: str) -> str:
    filename_ext = filename.split(".")[-1]
    filename_base = ".".join(filename.split(".")[:-1])
    return f"{filename_base}_{prefix}.{filename_ext}"

