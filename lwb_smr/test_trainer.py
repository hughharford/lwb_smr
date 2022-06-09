# Test File to run full pipeline on reduced dataset

import os

import tensorflow as tf
# from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import AUC, IoU

from lwb_smr.CustomDataLoader import CustomDataLoader
from lwb_smr.model import SMR_Model
from lwb_smr.utils_class.MLFlowPush import PushMLFlow
from lwb_smr.params import VM_path_dict
### test_trainer.py params
BATCH_SIZE = 32
EPOCHS = 1

EXPERIMENT_NAME = "UK Lon lwb_smr vertex_run_01"
EXPERIMENT_TAGS = {
    'USER': 'jhjhjhjh',
    'RUN NAME': 'test_trainner_check',
    'VERSION': '1.00',
    'DESCRIPTION': '''Model_02 UNET VGG-16, 1 epoch, 3 images'''
}

LOSS='binary_crossentropy'

class Test_Trainer():
    def __init__(self, VM=True):
        self.VM = VM
        self.loss = LOSS

    def just_get_the_data_loaded(self, X_path_key, y_path_key):
        """
        X_path_key: key for VM_path_dictionary - specifies X train, val or test data to load
        y_path_key: key for VM_path_dictionary - specifies y train, val or test data to load
        """
        if self.VM:
            x_images = os.listdir(VM_path_dict[X_path_key])
            x_path = VM_path_dict[X_path_key]

            y_masks = os.listdir(VM_path_dict[y_path_key])
            y_path = VM_path_dict[y_path_key]
        else:
            x_path = '/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/jpeg_train/'
            x_images = os.listdir(x_path) # raw rgb images

            y_path = '/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/jpeg_train/'
            y_masks = os.listdir(y_path) # mask to predict

        input_image_size = (250, 250)
        batch_size = 16
        customdata = CustomDataLoader(x_images, x_path, y_masks, y_path, input_image_size, batch_size)
        return customdata

    def set_model(self):

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
        return p

    def run(self):
        # Load in train, validation and test data.
        # Calls custom data laoder function, if VM = True, path will be taken from VM_path_dict
        # If VM = False will take path from local directory
        # NB VM=false not robust for local sources, only used for testing pipeline

        print(80*'-')
        print('------LOADING TRAIN DATA------')
        self.customdata_train = self.just_get_the_data_loaded('path_X_train', 'path_y_train') # arguments are the keys to VM_path_dict
        print(80*'.')
        print('------SUCCESS------')
        print(80*'-')
        print('------LOADING VALIDATION DATA------')
        self.customdata_val = self.just_get_the_data_loaded('path_X_val', 'path_y_val')
        print(80*'.')
        print('------SUCCESS------')

        # set mflow
        self.MFLOW = self.start_mlflow() # class instance of MLFLOW

        # set model
        self.set_model()

        mc = ModelCheckpoint('oxford_segmentation.h5', save_best_only=True) # could put path here
        # es = EarlyStopping()
        self.model.fit(
            self.customdata_train,
            self.customdata_val,
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
        print('------LOADING TEST DATA------')
        self.customdata_test = self.just_get_the_data_loaded('path_X_test', 'path_y_test')
        print(80*'.')
        print('------SUCCESS------')
        print(80*'-')
        print('------MODEL EVALUATING------')
        results = self.model.evaluate(self.customdata_test)


        self.MFLOW.mlflow_log_metric('loss', results[0])
        print(80*'=')
        print('------MODEL EVALUATED------')

if __name__ == '__main__':
    # print(os.getcwd())
    t = Test_Trainer(VM=False)
    # dataloader = t.just_get_the_data_loaded(VM=False)
    # print(type(dataloader))
    t.run()
    t.evaluate()

# /Users/jackhousego/code/hughharford/lwb_smr
