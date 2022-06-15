import pandas as pd
from lwb_smr.predict import PredictRoof



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

        print(79*'*')
        self.pred_roof = PredictRoof()
        self.pred_roof.tile_split(self.im_path_and_filename, 256, 256) # takes in the image_filename (but not the path)

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
        try:
            self.pred_roof.output_mask(self.roof_images)
            self.output_mask_path_and_filename = self.pred_roof.flashy_output()
        except Exception as e:
            print("PREDICTION ERROR: Something went wrong")
            print(e.__cause__)
            print(e.with_traceback)
            raise Exception("Sorry, no prediction available right now")
        finally:
            print("poorly coded, 'all error catching' ERROR CATCHING COMPLETED")

        # SET: self.output_mask_path_and_filename

        print((20*'_')+'DONE OUTPUT'+(20*'_'))
        return self.output_mask_path_and_filename

    def get_custom_roof_area(self,roof_num):
        '''
        user input to get roof area in m^2
        '''
        roof_area = self.pred_roof.get_roof_area(roof_num)
        return roof_area
