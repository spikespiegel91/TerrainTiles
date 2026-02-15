import mercantile
import rasterio
import rasterio.windows
from rasterio.windows import from_bounds
from PIL import Image
import numpy as np
import os

def get_elevation_decoder():
    """
    Returns the elevation decoder parameters for Mapbox Terrain RGB format.
    Use these parameters with deck.gl TerrainLayer or other terrain libraries.
    
    Returns:
        dict: Decoder parameters with rScaler, gScaler, bScaler, and offset
    """
    return {
        'rScaler': 256 * 256 * 0.1,    # Red channel multiplier
        'gScaler': 256 * 0.1,          # Green channel multiplier  
        'bScaler': 0.1,                # Blue channel multiplier
        'offset': -10000               # Height offset in meters
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
                if hasattr(src, 'nodata') and src.nodata is not None:
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
                if hasattr(src, 'nodata') and src.nodata is not None:
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

                        window = from_bounds(
                            bbox.west, bbox.south,
                            bbox.east, bbox.north,
                            src.transform
                        )

                        data = src.read(
                            1,
                            window=window,
                            out_shape=(256, 256),
                            resampling=rasterio.enums.Resampling.bilinear
                        )

                        # Fill NoData values with 0m (sea level) for consistent encoding
                        if hasattr(src, 'nodata') and src.nodata is not None:
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
                            img = Image.fromarray(rgb_data, mode='RGB')
                        else:
                            # Empty tile - sea level (0m) encoded
                            sea_level_encoded = int((0 + 10000) / 0.1)
                            r = (sea_level_encoded // (256 * 256)) % 256
                            g = (sea_level_encoded // 256) % 256
                            b = sea_level_encoded % 256
                            rgb_data = np.full((256, 256, 3), [r, g, b], dtype=np.uint8)
                            img = Image.fromarray(rgb_data, mode='RGB')

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
