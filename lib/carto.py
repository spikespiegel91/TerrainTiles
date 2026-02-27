from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds as rio_transform_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

from .io import *


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
        return parse_metadata_from_src(src)

def parse_metadata_from_src(src):
    return  {
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


def reproject_raster(
    filename: str,
    dirpath: str = "/Temp",
    output_path: str = "/Temp",
    dst_crs: str = "EPSG:3857",
    out_prefix: str | None = None,
    useBuffer: bool = False,
    num_threads: int = 2, #1
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
    out_memfile = None
    out_metadata = None


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
        if useBuffer:
            # Do NOT use MemoryFile as a context manager here — doing so would close it on exit,
            # making it impossible to open later. Only use `with` when opening it as a dataset.
            out_memfile = rasterio.MemoryFile(filename=out_filename)
            with out_memfile.open(**kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=rasterio.enums.Resampling.cubic,
                        num_threads=num_threads,
                    )
                
                out_metadata = parse_metadata_from_src(dst)
        
        else:
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
                        num_threads=num_threads,
                    )
                
                out_metadata = parse_metadata_from_src(dst)
                

    return out_path, out_filename, out_memfile, out_metadata


def reproject_raster2Mercator(
    filename: str,
    dirpath: str = "/Temp",
    output_path: str = "/Temp",
    useBuffer: bool = False,
    num_threads: int = 2, #1
):
    return reproject_raster(
        filename, dirpath, output_path, dst_crs="EPSG:3857", out_prefix="mercator_3857",
        useBuffer=useBuffer, num_threads=num_threads
    )


def reproject_raster2Geographic(
    filename: str,
    dirpath: str = "/Temp",
    output_path: str = "/Temp",
    useBuffer: bool = False,
    num_threads: int = 2, #1
):
    return reproject_raster(
        filename, dirpath, output_path, dst_crs="EPSG:4326", out_prefix="geo_4326",
        useBuffer=useBuffer, num_threads=num_threads
    )

