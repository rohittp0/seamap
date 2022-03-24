import pathlib

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


def create_images(dataset_path, data_variable):

    dataset_name = dataset_path.split('/')[-1].split('.')[0]

    pathlib.Path(f"data/{dataset_name}").mkdir(parents=True, exist_ok=True)
    data_raw = xr.open_dataset(dataset_path, decode_times=False)

    # data_raw = data_raw.get(data_variable).values

    for j in range(12):
        for k in range(57):
            numpy_array = data_raw.get("s_an").isel(
                depth=[k], time=[j]).values[0, 0]
            numpy_array[numpy_array == np.NAN] = -1

            numpy_array = numpy_array / (np.nanmax(numpy_array) / len(colors))

            image = np.ones((*numpy_array.shape, 3), dtype=np.uint8) * 255

            for i in range(len(colors)):
                image[(numpy_array >= i) & (numpy_array < i+1)] = colors[i]

            image = cv2.rotate(image, cv2.ROTATE_180)
            image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(f"data/{dataset_name}/{j}-{k}.png", image)


def main():
    create_images("netcdf/woa_salt.nc", "s_an")


if __name__ == '__main__':
    main()
