import pandas as pd
from lwb_smr.predict import PredictRoof

GOT_INPUT_IMAGE = False
LOADED_AND_READY = False
PREDICTED = False

class SolarMyRoof():
    def __init__(self):
        pass

    def load_and_ready(self, im_path_and_filename):
        '''
        instantiates PredictRoof
        loads the image and does the tile_split
        '''
        print(f'load_and_ready___: image_filename = {im_path_and_filename}')
        self.im_path_and_filename = im_path_and_filename
        self.GOT_INPUT_IMAGE = True

        print(79*'*')
        self.pred_roof = PredictRoof()
        self.pred_roof.tile_split(self.im_path_and_filename, 256, 256) # takes in the image_filename (but not the path)
        self.LOADED_AND_READY = True

        print((20*'_')+'DONE LOAD AND READY'+(20*'_'))


    def predict(self):
        '''
        loads the intended model
        perform_prediction to output roof_images
        '''

        print(79*'@')
        self.pred_roof.load_model()
        self.roof_images = self.pred_roof.perform_prediction()
        self.PREDICTED = True
        print((20*'_')+'DONE PREDICT'+(20*'_'))


    def output_completed_mask(self):
        '''
        if the following are done
            OUTPUT_MASK = False
        gets and returns output_mask image
        '''
        print(79*'$')
        if (self.GOT_INPUT_IMAGE + self.LOADED_AND_READY + self.PREDICTED) == 3:
            self.output_mask_path_and_filename = self.pred_roof.output_mask(self.roof_images)
        else:
            print(10 * " <<<< ERROR >>> ")

        # SET: self.output_mask_path_and_filename


        print((20*'_')+'DONE OUTPUT'+(20*'_'))
        return self.output_mask_path_and_filename
