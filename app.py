from flask import Flask
from flask import request

import xarray as xr

from plotter import get_plot

data_raw = xr.open_dataset("netcdf/woa_salt.nc", decode_times=False)

app = Flask(__name__, static_url_path='', static_folder="static/")


@app.route('/plot', methods=['POST', 'GET'])
def plot_route():
    type = request.args.get('type')

    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    time = int(request.args.get('time'))

    try:
        plot = get_plot(data_raw, "s_an", lat=lat, lon=lon, time=time)
    except Exception as e:
        print(e)
        plot = "No Data"

    return plot


if __name__ == '__main__':
    app.run()
