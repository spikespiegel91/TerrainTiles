"""
TerrainTiles Flask Application

A simple Flask API for terrain tiles manipulation.
"""
import time
from flask import Flask, request, jsonify, send_from_directory, g
import os
import json
import re

from matplotlib import pyplot
from mercantile import bounds

from lib import surface, tiles

from rasterio.windows import Window, from_bounds, shape, WindowMethodsMixin

import sqlite3
from lib.io import *

app = Flask(__name__)

# Configuration
TEMP_DIR = os.path.join(os.path.dirname(__file__), "Temp")
DEMOS_DIR = os.path.join(os.path.dirname(__file__), "demos")

# Ensure Temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)


def sanitize_filename(filename):
    """
    Sanitize filename to prevent path traversal attacks.

    Args:
        filename: The filename to sanitize

    Returns:
        A safe filename with only alphanumeric characters, underscores, and hyphens
    """
    # Remove any path separators and keep only safe characters
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", filename)
    # Ensure the filename is not empty after sanitization
    if not safe_name:
        safe_name = "unnamed"
    return safe_name


TILES_FOLDER = "Temp/tiles"


@app.route("/tiles")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tile Map</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
        <style>#map{height:100vh;}</style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map('map').setView([8, 13], 5);
            L.tileLayer('/tiles/{z}/{x}/{y}.png', {
                maxZoom: 20,
                minZoom: 5,
                tileSize: 256,
                noWrap: true,
            }).addTo(map);
        </script>
    </body>
    </html>
    """

@app.route("/tiles_db")
def index2():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tile Map</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
        <style>#map{height:100vh;}</style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map('map').setView([8, 13], 5);
            L.tileLayer('/tiles_db/{z}/{x}/{y}.png', {
                maxZoom: 20,
                minZoom: 5,
                tileSize: 256,
                noWrap: true,
            }).addTo(map);
        </script>
    </body>
    </html>
    """


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/tile", methods=["POST"])
def process_tile():
    """
    Process a terrain tile request.

    Expects JSON with:
    - name: tile name
    - coordinates: {lat, lon}
    - zoom_level: integer
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate required fields
        required_fields = ["name", "coordinates", "zoom_level"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Sanitize the filename to prevent path traversal
        safe_name = sanitize_filename(data["name"])

        # Save request to Temp directory for processing
        temp_file = os.path.join(TEMP_DIR, f"{safe_name}_request.json")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)

        # Process the tile (placeholder logic)
        result = {
            "status": "processed",
            "tile": data["name"],
            "coordinates": data["coordinates"],
            "zoom_level": data["zoom_level"],
            "temp_file": temp_file,
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/hello_plot")
def get_hello_plot():
    x, y, z = surface.get_grid_sample()
    surface.plot_contour(x, y, z)
    return


@app.route("/save_new_raster")
def save_new_raster():

    try:
        tempPath = os.path.join(TEMP_DIR, "new.tif")

        x, y, z = surface.get_grid_sample()
        # print(z)
        transform = surface.get_affine_transform(x, y)
        surface.save_new_raster(z, transform, tempPath)

        return jsonify({"status": "new raster created"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/show_new_raster")
def show_new_raster():
    try:
        # tempPath = os.path.join(TEMP_DIR, 'new.tif')
        # tempPath = os.path.join(DEMOS_DIR, 'venus2.tif')
        tempPath = os.path.join(DEMOS_DIR, "USGS_OPR_CA_SanFrancisco_B23_04300190.tif")

        surface.plot_raster(tempPath)

        return jsonify({"status": "raster displayed"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/raster_info")
def get_raster_info():
    try:
        # tempPath = os.path.join(TEMP_DIR, 'new.tif')
        # tempPath = os.path.join(DEMOS_DIR, 'venus2.tif')
        tempPath = os.path.join(DEMOS_DIR, "USGS_OPR_CA_SanFrancisco_B23_04300190.tif")
        raster_info = surface.get_raster_data(tempPath)

        return jsonify(raster_info), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_raster_tiles")
def get_raster_tiles():
    try:
        # tempPath = os.path.join(TEMP_DIR, 'new.tif')
        # tempPath = os.path.join(DEMOS_DIR, 'galaxy.tif')
        # tempPath = os.path.join(TEMP_DIR, 'venus2.tif')
        tempPath = os.path.join(DEMOS_DIR, "USGS_OPR_CA_SanFrancisco_B23_04300190.tif")

        output_dir = os.path.join(TEMP_DIR, "tiles")
        tiles.generate_tiles(tempPath, output_dir)

        return jsonify({"status": "tiles generated", "output_dir": output_dir}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/elevation_decoder")
def get_elevation_decoder():
    """Get elevation decoder parameters for terrain RGB tiles."""
    try:
        decoder_params = tiles.get_elevation_decoder()
        return (
            jsonify(
                {
                    "decoder": decoder_params,
                    "format": "Mapbox Terrain RGB",
                    "precision": "~1cm",
                    "range": "-10000m to +6553.5m",
                    "usage": "height = offset + (R * rScaler + G * gScaler + B * bScaler)",
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

################################################################################
@app.route("/tiles/<int:z>/<int:x>/<int:y>.png")
def serve_tile(z, x, y):
    """Serve individual terrain tiles."""
    try:
        tiles_dir = os.path.join(TEMP_DIR, "tiles")
        return send_from_directory(tiles_dir, f"{z}/{x}/{y}.png")
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    

################################################################################

def get_tiles_db():
    """Return a per-request SQLite connection, creating it once per request via Flask g."""
    if "tiles_db" not in g:
        g.tiles_db = sqlite3.connect(os.path.join(TEMP_DIR, "tiles.db"))
    return g.tiles_db

@app.teardown_appcontext
def close_tiles_db(exc):
    db = g.pop("tiles_db", None)
    if db is not None:
        db.close()

@app.route("/tiles_db/<int:z>/<int:x>/<int:y>.png")
def serve_tile_from_db(z, x, y):
    """Serve individual terrain tiles from database."""
    try:
        tileset = "tiles"  # "tiles_elevation" or "tiles_texture"
        db_cursor = get_tiles_db().cursor()
        return serve_image_from_tileset_db(db_cursor, tileset, x, y, z)
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    

################################################################################


@app.route("/generate_elevation_tiles")
def generate_elevation_tiles():
    try:
        _TEMP_DIR = os.path.join(os.path.dirname(__file__), "Temp")
        _DEMOS_DIR = os.path.join(os.path.dirname(__file__), "demos")

        # my_encoder = "terrain-rgb"
        # my_channels = 1
        settings = {
            "useBuffer": True,
            "num_threads": 1,
            "tiles_save_db": True,
            "tiles_db_name": "tiles.db", 
        }

        filename = "RGB.byte.tif"
        tileGenerator = tiles.TileGenerator(_DEMOS_DIR, filename, _TEMP_DIR, mode="tif", settings=settings)
        tileGenerator.set_tiles_name("tiles_elevation")

        tileGenerator.set_zoom([1, 2, 3])
        tileGenerator.generate_elevation_tiles()

        return jsonify({"status": "elevation tiles generated"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/generate_texture_tiles")
def generate_texture_tiles():
    try:
        _TEMP_DIR = os.path.join(os.path.dirname(__file__), "Temp")
        _DEMOS_DIR = os.path.join(os.path.dirname(__file__), "demos")

        settings = {
            "useBuffer": True,
            "num_threads": 1,  
            "tiles_save_db": True,
            "tiles_db_name": "tiles.db", 
        }

        filename = "RGB.byte.tif"
        tileGenerator = tiles.TileGenerator(_DEMOS_DIR, filename, _TEMP_DIR, mode="tif", settings=settings)
        tileGenerator.set_tiles_name("tiles_texture")

        # read from a .png texture
        # filename = "bani.png"
        # settings = { 
        #     "X_MIN": -70.46993753408448,
        #     "X_MAX": -70.39734766708911,
        #     "Y_MIN": 18.320038340921577,
        #     "Y_MAX": 18.37178547912261,
        #     "useBuffer": True,
        #     "num_threads": 1,  
        #     }

        # tileGenerator = tiles.TileGenerator(_DEMOS_DIR, filename, _TEMP_DIR, mode="png", settings = settings)

        tileGenerator.set_zoom([1, 2, 3])
        tileGenerator.generate_texture_tiles()

        return jsonify({"status": "texture tiles generated"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate_tiles")
def generate_tiles():

    try:
        _TEMP_DIR = os.path.join(os.path.dirname(__file__), "Temp")
        _DEMOS_DIR = os.path.join(os.path.dirname(__file__), "demos")

        my_encoder = None
        my_channels = None
        # my_channels = [1, 2, 3]

        my_encoder = "terrain-rgb"
        my_channels = 1
        settings = {
            "useBuffer": True,
            "num_threads": 1,  # 1
            "tiles_save_db": True,
            "tiles_db_name": "tiles.db", 
        }

        # my_encoder = "greyscale"
        # Z_MIN = 121.59725952148438
        # Z_RANGE = 249.82308959960938
        # HEIGHT_OFFSET =  1.02473
        # settings= { 
        #     "offset": Z_MIN,
        #     "rScaler": (1 / Z_RANGE) * 255, 
        #     "gScaler": 1, 
        #     "bScaler": 1
        # }

        tic = time.time()
        ##-------------------------------------------------

        filename = "RGB.byte.tif"
        # filename = "Bani.tif"
        tileGenerator = tiles.TileGenerator(_DEMOS_DIR, filename, _TEMP_DIR, mode="tif", settings=settings)
        tileGenerator.set_zoom([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12])
        # tileGenerator.set_zoom([5,12,13,14,15,16,17,18,19,20])
        # tileGenerator.set_zoom([5,12,13,14,15,16,17,18])
        # tileGenerator.set_zoom([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14])
        tileGenerator.save_tiles_png(indexes=my_channels, encoder=my_encoder, encoder_settings=settings)
        
        ##-------------------------------------------------
        toc = time.time()
        elapsed = toc - tic

        _tiledata = tileGenerator.get_tiles_data()
        _tilecount = tileGenerator.get_tiles_count()
        _src_metadata = tileGenerator.get_src_metadata()
        _out_metadata = tileGenerator.get_out_metadata()

        verbose = True
        if verbose:
            print(f"Tile data sample (first tile): {_tiledata[0] if _tiledata else 'No tile data'}")
            print(f"Source metadata: {_src_metadata}")
            print(f"Output metadata: {_out_metadata}")
            print(f"Total tiles generated: {_tilecount} in {elapsed:.2f} seconds")

        return (
            jsonify(
                {
                    "status": f"{_tilecount} tiles generated for {filename}",
                    # "src_metadata": _src_metadata,
                    # "out_metadata": _out_metadata,
                    # "tiledata": _tiledata,
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500





# FIXME: añadir esquema tiles/<int:z>/<int:x>/<int:y>.png' para poder servir de forma dinamica las tilas hacia el visor.
# @app.route('/tiles/dyn/<int:z>/<int:x>/<int:y>.png')
@app.route("/serve_image")
def serve_image_from_buffer(z, x, y):
    # tempPath = os.path.join(DEMOS_DIR, 'USGS_OPR_CA_SanFrancisco_B23_04300190.tif')
    # filename = 'USGS_OPR_CA_SanFrancisco_B23_04300190.tif'
    filename = "RGB.byte.tif"

    metada = tiles.get_raster_metadata(filename, DEMOS_DIR)
    # print('metada:', metada)
    print("metada.driver:", metada.get("driver", None))
    print("metada.crs:", metada.get("crs", None))
    print(
        "metada.width x height:",
        metada.get("width", None),
        "x",
        metada.get("height", None),
    )
    print("metada.bounds:", metada.get("bounds", None))
    print("metada.transform:", metada.get("transform", None))

    my_tile_shape = (256, 256)
    # my_tile_shape = None
    my_tiles_windows = None  # use to show the whole raster as a single tile
    #############################################################
    # # debug

    # left, bottom, right, top = metada['bounds']

    # x_width = right - left

    # Nsize = 2
    # step_diff = x_width / Nsize # use the same step for x and y to get a square tile

    # left_q1, bottom_q1, right_q1, top_q1 = left, bottom, left + step_diff, bottom + step_diff

    # q1_corners = (left_q1, bottom_q1, right_q1, top_q1)

    # tile_q1 = from_bounds(*q1_corners, metada['transform'])

    # my_tiles_windows = [tile_q1]
    #############################################################

    # data = tiles.read_tiledata_from_raster(filename, DEMOS_DIR, 1, tile_window=my_tiles_windows, out_shape=my_tile_shape)
    # # pyplot.imshow(data[0], cmap='pink')

    # img_data0 = tiles.data2img(data[0])
    # pyplot.imshow(img_data0, cmap='pink')
    # pyplot.show()

    # raster_m_path, filename_m = tiles.reproject_raster2Mercator(filename, DEMOS_DIR, TEMP_DIR)
    # raster_g_path, filename_g = tiles.reproject_raster2Geographic(filename, DEMOS_DIR, TEMP_DIR)
    # metada_g = tiles.get_raster_metadata(filename_g, TEMP_DIR)

    # metada_m = tiles.get_raster_metadata(filename_m, TEMP_DIR)

    # # bound_from_source_to_m = tiles.bounds2Mercator(metada['crs'], *metada['bounds'])

    # # necesito calcular la ventana para el sistema de coordenadas de mercator
    # bound_from_source_to_m = tiles.bounds2Mercator(metada['crs'], q1_corners)
    # # my_window_m = from_bounds(*bound_from_source_to_m, metada_m['transform'])
    # # esto podria ser una funcion
    # my_window_m = tiles.get_Mercator_window_from_bounds(metada['crs'], q1_corners, metada_m['transform'])

    # data_m = tiles.read_tiledata_from_raster(filename_m, TEMP_DIR, 1, tile_window=[my_window_m], out_shape=my_tile_shape)

    # print('metada_m.crs:', metada_m.get('crs', None ))
    # print('metada_m.bounds:', metada_m.get('bounds', None ))
    # print('metada_m.transform:', metada_m.get('transform', None ))

    # print('bound_from_source_to_m:', bound_from_source_to_m)

    # img_data0_m = tiles.data2img(data_m[0])
    # pyplot.imshow(img_data0_m, cmap='pink')
    # pyplot.show()

    ##############################################################################
    raster_g_path, filename_g = tiles.reproject_raster2Geographic(
        filename, DEMOS_DIR, TEMP_DIR
    )
    metada_g = tiles.get_raster_metadata(filename_g, TEMP_DIR)

    # ejemplo de busqueda y recorte de tilas dentro de un area de busqueda
    search_bounds = metada_g["bounds"]
    search_bounds_crs = metada_g["crs"]
    affine_coord2pixel = metada_g["transform"]

    tiles_inside = tiles.get_tiles(
        search_bounds, search_bounds_crs, zoom_levels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    my_window_m_ti = []

    for t in tiles_inside:
        print("tile:", t)
        geo_tile_bounds = t["geographic_bounds"]
        # bounds_ = (mercator_tile_bounds.left, mercator_tile_bounds.bottom, mercator_tile_bounds.right, mercator_tile_bounds.top)
        bounds_ = (
            geo_tile_bounds.west,
            geo_tile_bounds.south,
            geo_tile_bounds.east,
            geo_tile_bounds.north,
        )

        # win_ti = tiles.get_Mercator_window_from_bounds(metada['crs'], t['mercator_bounds'], metada_m['transform'])
        # win_ti = tiles.get_Mercator_window_from_bounds(metada_m['crs'], bounds_, metada_m['transform'])
        win_ti = tiles.get_Geographic_window_from_bounds(
            search_bounds_crs, bounds_, affine_coord2pixel
        )
        my_window_m_ti.append(win_ti)

    data_m_ti = tiles.read_tiledata_from_raster(
        filename_g, TEMP_DIR, 1, tile_window=my_window_m_ti, out_shape=my_tile_shape
    )

    # create a subplot for each element in data
    img_all = [tiles.data2img(data_) for data_ in data_m_ti]

    # img_data0_m_t0 = tiles.data2img(data_m_ti[0])
    img_data0_m_t1 = tiles.data2img(data_m_ti[1])

    # TODO vocar sobre disco las tilas con estructura z/x/y.png para poder servirlas de forma dinamica hacia el visor,
    ## crear mapa de indices

    for i, img in enumerate(img_all):
        if img is not None:

            pyplot.imshow(img, cmap="pink")
            pyplot.show()

    ########################################################################################

    # raster_g_path, filename_g = tiles.reproject_raster2Geographic(filename, DEMOS_DIR, TEMP_DIR)
    # metada_g = tiles.get_raster_metadata(filename_g, TEMP_DIR)

    # search_bounds = metada_g['bounds']
    # search_bounds_crs = metada_g['crs']
    # affine_coord2pixel = metada_g['transform']

    # # zoom =  f'{z}'
    # zoom =  z
    # tiles_inside = tiles.get_tiles(search_bounds, search_bounds_crs, zoom_levels =[zoom])
    # for t in tiles_inside:
    #     print('tile:', t)
    #     if t['z'] == zoom and t['x'] == x and t['y'] == y:
    #         print('found tile:', t)
    #         bounds_ = t['geographic_bounds']
    #         # geo_bounds =  t['geographic_bounds']
    #         # bounds_ = (geo_bounds.west, geo_bounds.south, geo_bounds.east, geo_bounds.north)
    #         win_ti = tiles.get_Geographic_window_from_bounds(search_bounds_crs, bounds_, affine_coord2pixel)

    #         dyn_data = tiles.read_tiledata_from_raster(filename_g, TEMP_DIR, 1, tile_window=[win_ti], out_shape=my_tile_shape)
    #         dyn_img = tiles.data2img(dyn_data[0])
    #         return tiles.serve_image_from_buffer(dyn_img)

    return tiles.serve_image_from_buffer(img_data0_m_t1)


if __name__ == "__main__":
    # WARNING: debug=True and host='0.0.0.0' should only be used for development.
    # For production, set debug=False and use a production WSGI server like gunicorn.
    app.run(debug=True, host="0.0.0.0", port=5000)
