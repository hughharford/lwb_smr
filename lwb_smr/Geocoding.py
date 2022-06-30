import googlemaps
import streamlit as st

class Geocoding():
    """
    A class to return the lattitude and longitude coordinates of a given address.
    Arguments: address --> [str]
    Return: geocode_dictionary --> [dict]
            {'lat': lattitude coordinate [float],
            'lng;: longitude coordinate [float],
            'boundary': lat and long for NE and SW boundary [dict],
            'searched_address': user entered search term [str],
            'formatted_address': full address [str]}
    """

    def __init__(self):
        APIKEY = st.secrets["GoogleMapsAPI"]
        self.map_client = googlemaps.Client(key=APIKEY)

    def get_geocoding(self, address):
        """
        Function will try to retrieve geocding information given the address paramater.
        If the address cannot be found, an exception is raised and a string returned.
        In such case, the string will cause the test in test_geocoding.py to fail.
        """
        self.address = address
        try:
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
        except:
            return "Address not found - ensure a correct address has been entered."


if __name__ == "__main__":
    test = Geocoding('SW1A 1AA')
    test_dict = test.get_geocoding()
    print(test_dict)
