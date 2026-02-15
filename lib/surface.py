import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from rasterio.transform import Affine

def hello_plot():
    fig, ax = plt.subplots()
    xdata = [1,2,3,4]
    ydata = [1,0,6,3]
    ax.plot(xdata, ydata)
    plt.show()

def get_grid_sample():
    x = np.linspace(-4.0, 4.0, 240)
    y = np.linspace(-3.0, 3.0, 180)[::-1]
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-2 * np.log(2) * ((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 1 ** 2)
    Z2 = np.exp(-3 * np.log(2) * ((X + 0.5) ** 2 + (Y + 0.5) ** 2) / 2.5 ** 2)
    Z = 10.0 * (Z2 - Z1)
    return x, y, Z  # Return 1D coordinate arrays and 2D data array
        
   
def plot_contour(x,y,z):
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    levels = np.linspace(np.min(z), np.max(z),10)
    ax.contour(X,Y,z, levels)

    plt.show()

# x,y,z = get_grid_sample()
# plot_contour(x,y,z)

def plot_3d_surface(x,y,z):
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, z, vmin=z.min() * 2, cmap=cm.Blues)

    ax.set(xticklabels=[],
        yticklabels=[],
        zticklabels=[])

    plt.show()

# x,y,z = get_grid_sample()
# plot_3d_surface(x,y,z)

def get_affine_transform(x, y):
    # x and y should be 1D coordinate arrays
    res_x = (x[-1] - x[0]) / (len(x) - 1)
    res_y = (y[-1] - y[0]) / (len(y) - 1)
    
    # Use the average resolution (they should be similar for square pixels)
    res = (abs(res_x) + abs(res_y)) / 2
    
    transform = Affine.translation(x[0] - res / 2, y[0] - res / 2) * Affine.scale(res, -res)
    
    return transform


def save_new_raster(Z, transform, output_path='Temp/new.tif'):
    # Convert Z to float32 if it's not already a compatible dtype
    if Z.dtype not in [np.float32, np.float64, np.int16, np.int32, np.uint8, np.uint16]:
        Z = Z.astype(np.float64)
    
    with rio.open(
        output_path,
        'w',
        driver='GTiff',
        height=Z.shape[0],
        width=Z.shape[1],
        count=1,
        dtype=Z.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(Z, 1)

def plot_raster(file_path):
    with rio.open(file_path) as src:
        Z = src.read(1)

        fig, ax = plt.subplots()
        ax.imshow(Z, cmap='terrain')
        plt.show()


def get_raster_data(file_path):
    with rio.open(file_path) as src:
        boundary = src.bounds
        transform = src.transform
        size = (src.width, src.height)
        xy_cgs = src.xy(size[1]//2, size[0]//2)
        # crs = src.crs

        return {
            'boundary': boundary,
            'transform': transform,
            'size': size,
            'center_coordinates': xy_cgs,
            #'crs': crs
        }
        

# x,y,z = get_grid_sample()

# transform = get_affine_transform(x,y)
# # print(transform)

# save_new_raster(z,transform)



