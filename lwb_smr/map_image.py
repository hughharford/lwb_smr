import googlemaps
from lwb_smr.params import predict_paths_dict
import os
APIKEY = ""

class GetMapImage():
    '''
    API to get 1280x1280 image from google maps
    to input into predict.py
    '''

    def __init__(self, address):
        self.address = address

    # def user_input(self):
    #     '''
    #     Ask the user to input an address
    #     '''
    #     self.address = input("Please enter a UK postcode:")

    def get_map(self):
        '''
        Process for retrieving map image
        '''
        # self.user_input()

        # get map
        map_client = googlemaps.Client(APIKEY)
        response = map_client.static_map(size=(1024,1024),scale=2,maptype ="satellite" ,center=(self.address),zoom= 17)

        # create filename
        im_name = self.address.upper().split()
        im_name = "_".join(im_name) +".jpg"
        self.filename = f"{predict_paths_dict['input_image']}{im_name}"

        # turn map get into saved image
        map_image = open(self.filename,"wb")
        for x in response:
            map_image.write(x)
        map_image.close()

        return im_name

    def remove_saved_file(self):
        os.remove(self.filename)
