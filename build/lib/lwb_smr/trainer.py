# THIS NEEDS TO BE WRITTEN OUT PROPERLY!
# ESPECIALLY THE FIRST BIT THAT SHOULD BE IN DATA.py as Josh has been writing
import os

import tensorflow as tf

from lwb_smr.CustomDataLoader import CustomDataLoader
from lwb_smr.model import SMR_Model
from lwb_smr.utils import PushMLFlow

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import AUC, IoU

BATCH_SIZE = 32
EPOCHS = 1
MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "UK Lon lwb_smr vertex_run_01"
EXPERIMENT_TAGS = {
    'USER': 'hsth',
    'RUN NAME': 'First vertex',
    'VERSION': 'M02_R01_xxx1',
    'DESCRIPTION': '''Model_02 UNET VGG-16, 1 epoch, full data'''
}

LOSS='binary_crossentropy'

class Trainer():
    def __init__(self):
        pass

    def just_get_the_data_loaded(self):
        # could be these (locally):
        # x_images = os.listdir('../raw_data/train_RGB_tiles/')
        # x_path = '../../raw_data/train_RGB_tiles/'
        x_images = os.listdir('../raw_data/AerialImageDataset/train/images/')
        x_path = '../raw_data/AerialImageDataset/train/images/'
        # y_masks = os.listdir('../../raw_data/train_mask_tiles/')
        # y_path = '../../raw_data/train_mask_tiles/'
        y_masks = os.listdir('../raw_data/AerialImageDataset/train/gt/')
        y_path = '../raw_data/AerialImageDataset/train/gt/'
        input_image_size = (250, 250)
        batch_size = 16
        customdata = CustomDataLoader(x_images, x_path, y_masks, y_path, input_image_size, batch_size)
        return customdata

    def set_model(self, loss=LOSS):
        self.loss = loss
        # Instantiate Model
        our_input_shape = (224,224,3)
        getVGG16 = SMR_Model(our_input_shape)
        self.model = getVGG16.get_latest_model()

        # Compile Model
        self.model.compile(
                    loss=self.loss,
                    optimizer=Adam(),
                    metrics=['accuracy'] #, tf.keras.metrics.AUC(), tf.keras.metrics.IoU()]
                    )

    def start_mlflow(self):
        p = PushMLFlow(EXPERIMENT_NAME, EXPERIMENT_TAGS)
        return p.mlflow_run

    def run(self):

        print(80*'-')
        print('------SETTING FOR DATA RUN------')

        customdata = self.just_get_the_data_loaded()

        print(80*'-')
        print('------MODEL RUNNING------')

        # set mflow
        self.MFLOW = self.start_mlflow()

        # set model
        self.set_model()

        mc = ModelCheckpoint('oxford_segmentation.h5', save_best_only=True) # could put path here
        # es = EarlyStopping()
        self.model.fit(
            customdata,
            # validation_split=0.3,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[mc]
            )

        self.MFLOW.mlflow_log_param('loss', self.loss)

        print(80*'=')
        print('------MODEL RUN SUCCESFULLY COMPLETED------')

        self.evaluate()

    def evaluate(self):
        print(80*'-')
        print('------MODEL EVALUATING------')
        results = self.model.evaluate(self.X_test, self.y_test)
        self.MFLOW.mlflow_log_metric('loss', results)
        print(80*'=')
        print('------MODEL EVALUATED------')

if __name__ == '__main__':
    t = Trainer()
    t.run()
