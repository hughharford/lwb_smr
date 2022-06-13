from params import PROJECT_ID, BUCKET_NAME
from params import BUCKET_FOLDER, BUCKET_MODEL_FOLDER
from params import BUCKET_TRAIN_DATA_FOLDER, BUCKET_TEST_DATA_FOLDER
from params import BUCKET_EE_DATA_OUTPUT

from utils import dice_loss

import tensorflow as tf

def GCP_Machine():
    def __init__(self):
        pass

    def get_model_from_gcp_bucket(self, model_name_inc_h5):
        path_to_models = BUCKET_NAME+ '/' + BUCKET_MODEL_FOLDER + '/'
        loading_model_gsutil = 'gs://' + path_to_models + model_name_inc_h5
        # '220611_VGG16_Dice_20e_in_shape_224x224x3.h5'

        custom_objects_dict = {
                        'dice_loss': dice_loss
        }
        loaded_model = tf.keras.models.load_model(loading_model_gsutil, custom_objects=custom_objects_dict)

        return loaded_model
