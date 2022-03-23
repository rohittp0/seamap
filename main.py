import time

import numpy as np
import xarray as xr
import cv2

LAT_RESOLUTION = 0.25
LON_RESOLUTION = 0.25
TIME_RESOLUTION = 1
DEPTH_RESOLUTION = 5

SCALE = 2

size = (1440 * SCALE, 720 * SCALE)


def main():
    data_raw = xr.open_dataset("data/woa_salt.nc", decode_times=False)

    numpy_array = data_raw.get("s_an").isel(depth=[10], time=[10]).values[0, 0]
    numpy_array = numpy_array / (np.nanmax(numpy_array) / 255)
    numpy_array = numpy_array.astype('uint8')

    image = cv2.cvtColor(numpy_array, cv2.COLOR_GRAY2RGB)
    image[:, :, 2] = 0
    image[:, :, 0] = 0
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

    cv2.imshow("Plot", image)
    cv2.waitKey()

    cv2.imwrite("plot.png", image)


if __name__ == '__main__':
    main()
