import requests
import polyline
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt


# Function get  route
def get_route(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):

    loc = "{},{};{},{}".format(
        pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    url = "http://62.106.95.167:5000/route/v1/driving/"
    r = requests.get(url + loc)
    if r.status_code != 200:
        return None

    res = r.json()
    route_coords = polyline.decode(res['routes'][0]['geometry'])

    return LineString(route_coords)


# Tehran Center
tehran_center_lon, tehran_center_lat = 51.411513, 35.711198

# Center and destination
pickup_lon, pickup_lat, dropoff_lon, dropoff_lat = tehran_center_lon, tehran_center_lat, 51.360702, 35.737955

# Calculate route
route = get_route(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)

# GeoDataFrame
gdf = gpd.GeoDataFrame(geometry=[route], crs='EPSG:4326')

print(gdf.total_bounds)

fig, ax = plt.subplots(figsize=(10, 6))

print(gdf.total_bounds)

gdf.plot(ax=ax, color='blue', linewidth=4)

ax.set_xlabel('Lon')
ax.set_ylabel('Lat')
plt.title('BiliRoute By OSRM')
plt.grid()
plt.show()
