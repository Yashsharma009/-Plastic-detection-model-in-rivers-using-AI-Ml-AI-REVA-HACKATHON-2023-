# Importing required module
from flask import Flask, render_template, request
import cv2
from ultralytics import YOLO
import os
import folium
from werkzeug.utils import secure_filename
import ee

app = Flask(__name__)

def convert_to_degrees(value):
    d = value[0]
    m = value[1]
    s = value[2]
    return d + (m / 60.0) + (s / 3600.0)

def get_gps_coordinates(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        geotags = {}
        for tag in tags.keys():
            if tag.startswith('GPS'):
                geotags[tag] = tags[tag]

        latitude = geotags.get('GPS GPSLatitude')
        longitude = geotags.get('GPS GPSLongitude')

        if latitude and longitude:
            latitude = convert_to_degrees(latitude.values)
            longitude = convert_to_degrees(longitude.values)

            if 'S' in str(geotags.get('GPS GPSLatitudeRef')):
                latitude = -latitude
            if 'W' in str(geotags.get('GPS GPSLongitudeRef')):
                longitude = -longitude

            return latitude, longitude

    except Exception as e:
        raise ValueError(f"Error while extracting GPS information from {image_path}") from e

    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded image file
    image = request.files['image']

    # Save the image to a folder
    image.save('static/uploads/' + image.filename)
    model = YOLO('best.pt')
    model.predict('static/uploads/' + image.filename, save=True)
    img = cv2.imread('runs/detect/predict/' + image.filename)
    cv2.imwrite('static/results/' + image.filename, img)
    # Define the image_name variable
    image_name = image.filename

    import shutil
    # Specify the path of the folder to be deleted
    folder_path = 'runs'

    # Remove the folder and its contents
    shutil.rmtree(folder_path)

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            # Save the uploaded file
            file_path = os.path.join('uploads', secure_filename(file.filename))
            file.save(file_path)

            # Get GPS coordinates and other information
            coordinates = get_gps_coordinates(file_path)

            if coordinates:
                latitude, longitude = coordinates

                # Set up Earth Engine
                ee.Initialize()

                # Create a point of interest as an ee.Geometry
                poi = ee.Geometry.Point([longitude, latitude])

                # Get Sentinel-2 image collection
                sentinel = ee.ImageCollection("COPERNICUS/S2_SR") \
                    .filterBounds(poi) \
                    .filterDate('2022-01-01', '2022-12-31') \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

                # Calculate NDVI for each image in the collection
                def calculate_ndvi(image):
                    ndvi = image.normalizedDifference(['B5', 'B4'])
                    return image.addBands(ndvi.rename('NDVI'))

                sentinel_with_ndvi = sentinel.map(calculate_ndvi)

                # Select the first image in the collection
                first_image = ee.Image(sentinel_with_ndvi.first())

                # Get the NDVI band
                ndvi_band = first_image.select('NDVI')

                # Get water quality parameters using Sentinel-2 bands
                water_quality_params = first_image.select(['B3', 'B4', 'B8A'])

                # Calculate NDWI
                ndwi = first_image.normalizedDifference(['B3', 'B8A'])
                ndwi_mean = ndwi.reduceRegion(ee.Reducer.mean(), poi).get('nd').getInfo()

                # Determine water quality based on NDWI
                if ndwi_mean < 0.2:
                    water_quality = "Water quality is good."
                elif 0.2 <= ndwi_mean < 0.5:
                    water_quality = "Water quality is moderate."
                else:
                    water_quality = "Water quality is poor. Action may be needed."

                # Create a folium map with specified dimensions
                map_center = [latitude, longitude]
                map_zoom = 15
                map_width = 500
                map_height = 500

                mymap = folium.Map(location=map_center, zoom_start=map_zoom, width=map_width, height=map_height)

                # Add a marker for the image location
                folium.Marker(location=map_center, popup="Image Location").add_to(mymap)

                # Display the image on the map using folium.ImageOverlay
                image_url = f'https://earthengine.googleapis.com/v1alpha/projects/earthengine-legacy/thumbnails/{first_image.getMapId()}?token={first_image.getMapId()}'
                image_overlay = folium.raster_layers.ImageOverlay(
                    image_url,
                    bounds=[[latitude, longitude]],
                    opacity=0.6,
                ).add_to(mymap)

                # Save the map as an HTML file
                map_html_path = os.path.join('templates', 'map.html')
                mymap.save(map_html_path)

                # Render the result page with map and information
                return render_template('result.html', image_path=file_path, map_html_path='map.html',
                                       ndvi=ndvi_band.reduceRegion(ee.Reducer.mean(), poi).get('NDVI').getInfo(),
                                       water_quality_params=water_quality_params.reduceRegion(ee.Reducer.mean(),
                                                                                                poi).getInfo(),
                                       ndwi=ndwi_mean, water_quality=water_quality)

    return render_template('display.html', image_name=image_name)

if __name__ == '__main__':
    app.run(debug=True)
