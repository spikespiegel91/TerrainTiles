import rasterio
from PIL import Image
import os
from numpy import ndarray
from io import BytesIO
from flask import send_file
import sqlite3


def serve_image_from_buffer(img: Image.Image):
    img_io = BytesIO()
    img.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

def get_image_bytes(img: Image.Image) -> bytes:
    img_io = BytesIO()
    img.save(img_io, "PNG")
    img_io.seek(0)
    return img_io.getvalue()


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


def save_tile_directory(img:Image.Image, basedir:str,x:int, y:int, z:int):
    path = f"{basedir}/{z}/{x}"
    os.makedirs(path, exist_ok=True)
    img.save(f"{path}/{y}.png")

#TODO: add a new method to save the tiles into an sqlite database (e.g., MBTiles format)
# do I need to pass the path to the db or just the connection? 
# it is silly to connect the db every time I save a tile !!
def save_tile_db(img:Image.Image, db_cursor: any, tileset:str ,x:int, y:int, z:int):
    # Convert the image to bytes

    img_bytes = get_image_bytes(img)
    sqlite_blob = sqlite3.Binary(img_bytes)


    # Insert the tile into the database
    db_cursor.execute(
        "INSERT INTO tiles (tileset, zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?, ?)",
        (tileset, z, x, y, sqlite_blob)
    )

    # db_cursor.execute(
    #             "INSERT OR REPLACE INTO tiles "
    #             "(zoom_level, tile_column, tile_row, tile_data) "
    #             "VALUES (?, ?, ?, ?);",
    #             (tile.z, tile.x, tiley, sqlite3.Binary(contents)),

    #         )

def delete_tileset_from_db(db_cursor: any, tileset:str):
    try:
        # first check if the tileset exists in the database, if there is no such tileset,
        #  we can skip the deletion

        db_cursor.execute(
            "SELECT COUNT(*) FROM tiles WHERE tileset = ?",
            (tileset,)        )
        count = db_cursor.fetchone()[0] 

        if count == 0:
            print(f"No tiles found for tileset {tileset} in database, skipping deletion.")
            return


        db_cursor.execute(
            "DELETE FROM tiles WHERE tileset = ?",
            (tileset,)
        )
    except Exception as e:
        print(f"Error deleting tileset {tileset} from database: {e}")

def serve_image_from_tileset_db(db_cursor: any, tileset: str, x: int, y: int, z: int):
    db_cursor.execute(
        "SELECT tile_data FROM tiles WHERE tileset = ? AND zoom_level = ? AND tile_column = ? AND tile_row = ?",
        (tileset, z, x, y)
    )
    result = db_cursor.fetchone()
    if result is None:
        raise ValueError(f"Tile not found in database for tileset: {tileset}, zoom: {z}, x: {x}, y: {y}")
    
    img_bytes = result[0]
    return send_file(BytesIO(img_bytes), mimetype="image/png")
    # img_io = BytesIO(img_bytes)
    # img_io.seek(0)
    # return send_file(img_io, mimetype="image/png")