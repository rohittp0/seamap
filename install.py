import numpy as np
import xarray as xr
import cv2


LAT_RESOLUTION = 0.25
LON_RESOLUTION = 0.25
TIME_RESOLUTION = 1
DEPTH_RESOLUTION = 5

SCALE = 2

size = (1440 * SCALE, 720 * SCALE)

colors = [[148, 0, 211], [75, 0, 130], [0, 0, 255], [0, 255, 0],
          [255, 255, 0], [255, 127, 0], [255, 0, 0]]


def main():
    data_raw = xr.open_dataset("data/woa_salt.nc", decode_times=False)

    numpy_array = data_raw.get("s_an").isel(depth=[0], time=[11]).values[0, 0]
    numpy_array[numpy_array == np.NAN] = -1

    numpy_array = numpy_array / (np.nanmax(numpy_array) / len(colors))

    image = np.ones((*numpy_array.shape, 3), dtype=np.uint8) * 255

    for i in range(len(colors)):
        image[(numpy_array >= i) & (numpy_array < i+1)] = colors[i]

    image = cv2.rotate(image, cv2.ROTATE_180)
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite("plot.png", image)


if __name__ == '__main__':
    main()
