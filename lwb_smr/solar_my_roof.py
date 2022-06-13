import pandas as pd
from lwb_smr.predict import PredictRoof


class SolarMyRoof():
    def __init__():
        pass

    def grab_input_image(self, postcode='None', borough_name='None'):
        if postcode != 'None':
            self.postcode = postcode
        if borough_name != 'None':
            self.borough_name
        self.load_and_ready(self.postcode)

    def load_and_ready(self, postcode='None'):
        print(79*'*')
        self.pred_roof = PredictRoof()
        if postcode == 'None':
            self.pred_roof.tile_split()
        elif postcode != 'None':
            self.pred_roof.tile_split(self.postcode)

        print((20*'_')+'DONE LOAD AND READY'+(20*'_'))


    def predict(self):
        print(79*'@')
        self.pred_roof.load_model()
        self.roof_images = self.pred_roof.perform_prediction()
        print((20*'_')+'DONE PREDICT'+(20*'_'))


    def output_completed_work(self):
        print(79*'$')
        self.pred_roof.output_mask(self.roof_images)
        print((20*'_')+'DONE OUTPUT'+(20*'_'))
        return self.pred_roof.output_mask
