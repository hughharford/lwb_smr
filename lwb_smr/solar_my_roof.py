import pandas as pd
from lwb_smr.predict import PredictRoof

GOT_INPUT_IMAGE = False
LOADED_AND_READY = False
PREDICTED = False

class SolarMyRoof():
    def __init__():
        pass

    def grab_input_image(self, postcode='None', borough_name='None'):
        '''
        takes input postcode, if there is one,
        then calls load_and_ready
        '''

        if postcode != 'None':
            self.postcode = postcode
        if borough_name != 'None':
            self.borough_name

        GOT_INPUT_IMAGE = True
        self.load_and_ready(self.postcode)

    def load_and_ready(self, postcode='None'):
        '''
        instantiates PredictRoof
        does the tile_split
        '''
        print(79*'*')
        self.pred_roof = PredictRoof()

        if postcode == 'None':
            self.pred_roof.tile_split() # takes in the image_filename (but not the path)
        elif postcode != 'None':
            self.pred_roof.tile_split()

        LOADED_DATA = False
        print((20*'_')+'DONE LOAD AND READY'+(20*'_'))


    def predict(self):
        '''
        loads the intended model
        perform_prediction to output roof_images
        '''

        print(79*'@')

        self.pred_roof.load_model()
        self.roof_images = self.pred_roof.perform_prediction()
        PREDICTED = False
        print((20*'_')+'DONE PREDICT'+(20*'_'))


    def output_completed_work(self):
        '''
        if the following are done
            OUTPUT_MASK = False
        gets and returns output_mask image
        '''
        print(79*'$')
        if sum(GOT_INPUT_IMAGE, LOADED_AND_READY, PREDICTED) == 3:
            pass
        self.pred_roof.output_mask(self.roof_images)
        print((20*'_')+'DONE OUTPUT'+(20*'_'))
        return self.pred_roof.output_mask
