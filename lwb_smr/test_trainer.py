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
from lwb_smr.params import VM_path_dict, csv_path_dict
from lwb_smr.data import GetData, LoadDataSets

### DATALOADER PARAMETERS
INPUT_IMAGE_SIZE = (250, 250) # add to global variables
BATCH_SIZE = 8

### MLFLOW PARAMETERS
EXPERIMENT_NAME = "UK Lon lwb_smr vertex_run_01"
EXPERIMENT_TAGS = {
    'USER': 'jhjhjhjh',
    'RUN NAME': 'test_trainner_check',
    'VERSION': '1.00',
    'DESCRIPTION': '''Model_02 UNET VGG-16, 1 epoch, 3 images'''
}

### MODEL PARAMETERS
LOSS='binary_crossentropy'
OUR_INPUT_SHAPE = (224, 224, 3)
METRICS = ['accuracy']
EPOCHS = 2

class Test_Trainer():
    def __init__(self, VM=True):
        self.VM = VM
        self.loss = LOSS
        # Load dictionary containing all images and mask file names in corresponding
        # csv file:
        self.data_dict = LoadDataSets(csv_path_dict['train_csv'],
                                         csv_path_dict['val_csv'],
                                         csv_path_dict['test_csv']).load_datasets()

    def just_get_the_data_loaded(self, X_key, y_key):
        """
        X_path_key: key for VM_path_dictionary - specifies X train, val or test data to load
        y_path_key: key for VM_path_dictionary - specifies y train, val or test data to load

        Returns: customdata loader class object

        Description: Call this function for each train, val and test specifiying the key to data_dictionary
                     to access the list of image and mask file names & path to folder containing images.
        """

        # If using Virtual Machine Load data from VM paths otherwise local path
        if self.VM:
            # Load filenames e.g. ['austin_x00_y00.jpeg', 'austin_x00_y01.jpeg' ...]
            x_images = self.data_dict[X_key][:2*BATCH_SIZE]
            # Load path to folder containing images
            x_path = VM_path_dict['path_x']

            # Load filenames e.g. ['austin_x00_y00_mask.jpeg', 'austin_x00_y01_mask.jpeg' ...]
            y_masks = self.data_dict[y_key][:2*BATCH_SIZE]
            # Load path to folder containing masks
            y_path = VM_path_dict['path_y']
        else:
            # For running file on a local machine
            x_path = '/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/jpeg_train/'
            x_images = os.listdir(x_path) # raw rgb images

            y_path = '/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/jpeg_train/'
            y_masks = os.listdir(y_path) # mask to predict


        customdata = CustomDataLoader(x_images, x_path, y_masks, y_path, INPUT_IMAGE_SIZE, BATCH_SIZE)
        return customdata

    def set_model(self):

        # Instantiate Model
        getVGG16 = SMR_Model(OUR_INPUT_SHAPE)
        self.model = getVGG16.get_latest_model()

        # Compile Model
        self.model.compile(
                    loss=self.loss,
                    optimizer=Adam(),
                    metrics=METRICS #, # more useful to see what metrics are here?? tf.keras.metrics.AUC(), tf.keras.metrics.IoU()]
                    )

    def start_mlflow(self):
        p = PushMLFlow(EXPERIMENT_NAME, EXPERIMENT_TAGS)
        return p # returns a class instance of PushMLFlow

    def run(self):
        # Load in train, validation and test data.
        # Calls custom data laoder function, if VM = True, path will be taken from VM_path_dict
        # If VM = False will take path from local directory
        # NB VM=false not robust for local sources, only used for testing pipeline

        print(80*'-')
        print('------LOADING TRAIN DATA------')
        self.customdata_train = self.just_get_the_data_loaded(X_key='train_x', y_key='train_y') # arguments are the keys to VM_path_dict
        print(80*'.')
        print('------SUCCESS------')
        print(80*'-')
        print('------LOADING VALIDATION DATA------')
        self.customdata_val = self.just_get_the_data_loaded(X_key='val_x', y_key='val_y') # dataloader object
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
            # batch_size=BATCH_SIZE, # potential to take this line out (batch size defined in dataloader)
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
        self.customdata_test = self.just_get_the_data_loaded(X_key='test_x', y_key='test_y')
        print(80*'.')
        print('------SUCCESS------')
        print(80*'-')
        print('------MODEL EVALUATING------')
        results = self.model.evaluate(self.customdata_test)


        self.MFLOW.mlflow_log_metric('loss', results[0])
        print(80*'=')
        print('------MODEL EVALUATED------')

if __name__ == '__main__':
    t = Test_Trainer(VM=False)
    t.run()
    t.evaluate()
