import json
import morecantile
import os
import rasterio
import requests

BASE_URL = "https://d1nzvsko7rbono.cloudfront.net"
BASE_TILE_URL = "{BASE_URL}/mosaic/tiles/{searchid}/WebMercatorQuad/{z}/{x}/{y}.tif"

REGISTER_ENDPOINT = f"{BASE_URL}/mosaic/register"

TILE_URL = {
    "HLSL30": f"{BASE_TILE_URL}?assets=B02&assets=B03&assets=B04&assets=B05&assets=B06&assets=B07",
    "HLSS30": f"{BASE_TILE_URL}?assets=B02&assets=B03&assets=B04&assets=B8A&assets=B11&assets=B12",
}

PROJECTION = "WebMercatorQuad"
TMS = morecantile.tms.get(PROJECTION)
ZOOM_LEVEL = 12
DOWNLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "../data")


class Downloader:
    def __init__(self, date, layer="HLSL30"):
        """
        Initialize Downloader
        Args:
            date (str): Date in the format of 'yyyy-mm-dd'
            layer (str): any of HLSL30, HLSS30
        """
        self.layer = layer
        self.date = date
        self.search_id = self.register_new_search()
        pass

    def download_tile(self, x_index, y_index, filename):
        return_filename = filename
        response = requests.get(
            TILE_URL[self.layer].format(
                BASE_URL=BASE_URL,
                searchid=self.search_id,
                z=ZOOM_LEVEL,
                x=x_index,
                y=y_index,
            )
        )
        if response.status_code == 200:
            with open(filename, "wb") as download_file:
                download_file.write(response.content)
            raster_file = rasterio.open(filename)
            profile = raster_file.profile
            profile['dtype'] = 'float32'
            with rasterio.open(filename, 'w', **profile) as new_file:
                for band in range(profile['count']):
                    index = band + 1
                    new_file.write(raster_file.read(index) * 0.0001, index)
            raster_file.close()
        else:
            return_filename = ""
        return return_filename

    def mkdir(self, foldername):
        if not (os.path.exists(foldername)):
            os.makedirs(foldername)

    def download_tiles(self, bounding_box):
        x_tiles, y_tiles = self.tile_indices(bounding_box)
        downloaded_files = list()
        for x_index in range(x_tiles[0], x_tiles[1]):
            for y_index in range(y_tiles[0], y_tiles[1]):
                self.mkdir(f"{DOWNLOAD_FOLDER}/{self.layer}")
                filename = f"{DOWNLOAD_FOLDER}/{self.layer}/{self.date}-{x_index}-{y_index}.tif"
                if os.path.exists(filename):
                    downloaded_files.append(filename)
                else:
                    downloaded_file = self.download_tile(x_index, y_index, filename)
                    downloaded_files.append(downloaded_file) if downloaded_file else ""
        return downloaded_files

    def register_new_search(self):
        """
            Register new search with HLS titiler
        Args:
            date (str): Date in the format of 'yyyy-mm-dd'
        """
        response = requests.post(
            REGISTER_ENDPOINT,
            headers={"Content-Type": "application/json", "accept": "application/json"},
            data=json.dumps(
                {
                    "datetime": f"{self.date}T00:00:00Z/{self.date}T23:59:59Z",
                    "collections": [self.layer],
                }
            ),
        ).json()
        return response["searchid"]

    def tile_indices(self, bounding_box):
        """
            Extract tile indices based on bounding_box

        Args:
            bounding_box (list): [left, down, right, top]

        Returns:
            list: [[start_x, end_x], [start_y, end_y]]
        """
        start_x, start_y, _ = TMS.tile(bounding_box[0], bounding_box[3], ZOOM_LEVEL)
        end_x, end_y, _ = TMS.tile(bounding_box[2], bounding_box[1], ZOOM_LEVEL)
        return [[start_x, end_x], [start_y, end_y]]
