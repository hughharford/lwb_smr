import math
from statistics import mean
import streamlit as st
from streamlit_folium import folium_static
import folium
import pandas as pd
import datetime

'''
# TaxiFareModel front
'''

class StateTracker():
    def __init__(self):
        self.state_pickup_lat = 0
        self.state_pickup_lon = 0

        self.state_dropoff_lat = 0
        self.state_dropoff_lon = 0

tracker = StateTracker()

centralCoordinates = [40.767937,-73.982155]
# m = folium.Map(location=centralCoordinates, zoom_start=3)
m = folium.Map(location=centralCoordinates,tiles='OpenStreetMap', zoom_start=10)
folium_static(m)

# Use a callback to display the current value of the slider when changed
def dropoff_onto_map():
    st.write("The value of the slider is:", st.session_state.myslider)
    # add drop off
    folium.Marker(
        location=[dropoff_latitude, dropoff_longitude],
        popup="Your dropoff",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

def pickup_onto_map():
    # add pickup
    folium.Marker(
        location=[pickup_latitude, pickup_longitude],
        popup="Your pickup",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)




def pickup_both_updated():
    pass

def dropoff_both_updated():
    pass

def pickup_lat_updated(tracker):
    tracker.state_pickup_lat = 1

def pickup_lon_updated(tracker):
    tracker.state_pickup_lon = 1

def dropoff_lat_updated(tracker):
    tracker.state_dropoff_lat = 1

def dropoff_lon_updated(tracker):
    tracker.state_dropoff_lon = 1


# - date and time
formatted_pickup_datetime = st.time_input('Input your date and time for Pickup', datetime.time(8, 45))
st.write('Pickup is set for', formatted_pickup_datetime, ' today')
# st.write(type(formatted_pickup_datetime))

st.header('''CHOOSE PICKUP:''')

long_lat_step = 0.01
#~~~~~~~~~~~~~~~~~~~~
nyc_lat_min = 40.72
nyc_lat_max = 40.81
start_lat = mean([nyc_lat_min,nyc_lat_max])
nyc_lon_min = -74.10
nyc_lon_max = -73.75



# st.slider(
#     "My Slider", 0.0, 100.0, 1.0, step=1.0, key="myslider", on_change=dropoff_onto_map
# )

# latitude, longitude, e.g. Empire State Building lies at latitude 40.748440°, longitude -73.984559°
# - pickup latitude
pickup_latitude = st.slider('Please choose a latitude for PICKUP ',
                            nyc_lat_min, nyc_lat_max, step=long_lat_step,
                            value=start_lat,
                            on_change=pickup_both_updated)
# st.write("PICKUP latitude: ", pickup_latitude, '')
# - pickup longitude
pickup_longitude = st.slider('Please choose a longitude for PICKUP ',
                             nyc_lon_min, nyc_lon_max,
                             step=long_lat_step,
                             value=start_lat,
                             on_change=pickup_both_updated)
# st.write("PICKUP longitude: ", pickup_longitude, '')


st.header('''CHOOSE DROPOFF:''')

# - dropoff_latitude
dropoff_latitude = st.slider('Please choose a latitude for DROPOFF ', nyc_lat_min, nyc_lat_max, step=long_lat_step)
# st.write("DROPOFF latitude: ", dropoff_latitude, '')
# - dropoff_longitude
dropoff_longitude = st.slider('Please choose a longitude for DROPOFF ', nyc_lon_min, nyc_lon_max, step=long_lat_step)
# st.write("DROPOFF longitude: ", dropoff_longitude, '')

'''

'''
# st.header("you chose: ")
# st.write("PICKUP longitude: ", pickup_longitude, '')
# st.write("PICKUP latitude: ", pickup_latitude, '')
# st.write("DROPOFF longitude: ", dropoff_longitude, '')
# st.write("DROPOFF latitude: ", dropoff_latitude, '')









# not possible to demo this without setting the full site in wide mode

# st.code("")


params = {'pickup_datetime': "2013-07-06 17:18:00",
          'pickup_longitude': pickup_longitude,
          'pickup_latitude': pickup_latitude,
          'dropoff_longitude': dropoff_longitude,
          'dropoff_latitude': dropoff_latitude,
          'passenger_count': 1
        }

query_string = '/?pickup_datetime=2012-10-06%2012:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2'

import requests
url = 'https://fastapi-dkljdrdtqa-ew.a.run.app/predict'
#'https://taxifare.lewagon.ai/predict' # USE response['fare']

response = requests.get(url, params=params).json()
# st.write(response.url)
st.write(round(response['prediction'],2))
