import googlemaps
import streamlit as st

class Geocoding():

    def __init__(self, address):
        self.address = address
        APIKEY = st.secrets["GoogleMapsAPI"]
        self.map_client = googlemaps.Client(key=APIKEY)

    def get_geocoding(self):

        geocode_result = self.map_client.geocode(self.address)

        geo_dict = {
            'lat': geocode_result[0]['geometry']['location']['lat'],
            'lng': geocode_result[0]['geometry']['location']['lng'],
            'boundary': {
                'northeast': geocode_result[0]['geometry']['bounds']['northeast'],
                'southwest': geocode_result[0]['geometry']['bounds']['southwest'],
            },
            'searched_address': self.address,
            'formatted_address': geocode_result[0]['formatted_address']
        }

        return geo_dict


if __name__ == "__main__":
    test = Geocoding('SW1A 1AA')
    test_dict = test.get_geocoding()
    print(test_dict)
