import googlemaps
# <<<<<<< day7_jack
# from lwb_smr.params import predict_paths_dict
# import os
# APIKEY = ""
# >>>>>>>>>>>> WHERE YOU DOING THE API KEY MANUALLY?
# =======
from lwb_smr.params import predict_paths_dict, prediction_path_dict
from env import GoogleMapsAPI
APIKEY = GoogleMapsAPI

class GetMapImage():
    '''
    API to get 1280x1280 image from google maps
    to input into predict.py
    '''

    def __init__(self, input_location):
        self.address = input_location

    def user_input(self):
        '''
        Ask the user to input an address.
        Optional, not currently expected to use this...
        '''
        self.address = input("Please enter a UK postcode:")

    def get_map(self):
        '''
        Process for retrieving map image
        '''

        #  self.user_input() # this input is given on instantiation of the class
        # see self.address in __init__()

        # get map
        map_client = googlemaps.Client(APIKEY)
        response = map_client.static_map(size=(1024,1024),scale=2,maptype ="satellite" ,center=(self.address),zoom= 17)

        # create path and filename
        im_name = self.address.upper().split()
        im_name = "_".join(im_name) +".jpg"
        im_path_and_filename = f"{prediction_path_dict['all_files_here']}{im_name}"

        # for test only:
        print(f'im_path_and_filename = {self.im_path_and_filename}')

        # turn map get into saved image
        map_image = open(self.im_path_and_filename,"wb")
        for x in response:
            map_image.write(x)
        map_image.close()


        return self.im_path_and_filename

      def remove_saved_file(self):
          os.remove(self.im_path_and_filename)
