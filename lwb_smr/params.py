"""
File to store parameters for reuse

PLEASE adapt experiment tags appropriately.
initials especially
model name to cover epochs, architecture
REMOVE test_ when doing a proper run

AND, of course, set TEST_RUN to false, to do an actual run

"""
# 24 07 11: model for testing:
string_test_model_name = "models_220615_tbc__v7_UNET_VGG16_Dice_input_shape_224x224x3.h5"
nope = "220611_v7_UNET_VGG16_Dice_input_shape_224x224x3.h5"
string_demo_model_name = "220622_model_selected.h5"

MODEL_FOR_PREDICTION = string_test_model_name  # must reside in the path:
#                           lwb_smr/lwb_smr/data/demo_files/prediction


# NB: this sets variables below, so only set this to True/False
TEST_RUN = True
DATA_AUGMENTATION = True

TEST_BATCH_SIZE = 8
TEST_EPOCHS = 2

RUN_BATCH_SIZE = 60
RUN_EPOCHS = 30

if TEST_RUN:
    BATCH_SIZE = TEST_BATCH_SIZE
    EPOCHS = TEST_EPOCHS
else:
    BATCH_SIZE = RUN_BATCH_SIZE
    EPOCHS = RUN_EPOCHS

UNET_INPUT_SHAPE = (224,224,3)
IMAGE_SQ_SIZE = 224
possibleLOSS=['binary_crossentropy','DICE', 'custom_combo']
LOSS = possibleLOSS[2]

Experiment_name_base = 'UK Lon lwb_smr'
EXPERIMENT_NAME = Experiment_name_base + " test_run_02" # template
EXPERIMENT_TAGS = {
    'USER': 'hsth_day10',
    'RUN NAME': 'TEST_evaluation of models',
    'VERSION': 'M2_R04_15',
    'DESCRIPTION': 'TEST_Models x2',
    'LOSS': LOSS,
    'METRICS': 'accuracy, binaryIoU, AUC'
}


PROJECT_ID = 'lwb-solar-my-roof'
# le-wagon-bootcamp-347615
## cannot set PROJECT (with set_project) to:
#
IMAGE='tbc___XXX'
REGION='europe-west1'
MULTI_REGION='eu.gcr.io'
BUCKET_NAME='lwb-solar-my-roof'

# BUCKET FOLDERS
BUCKET_FOLDER='data'
BUCKET_TRAIN_DATA_FOLDER='train'
BUCKET_TEST_DATA_FOLDER='test'
BUCKET_MODEL_FOLDER = 'models'
BUCKET_DEMO_FILES_FOLDER = 'demo_files'
# BUCKET_CHECKPOINT_FOLDER = checkpoints
BUCKET_EE_DATA_OUTPUT = 'ee-data-output'


# MLFlow URI to save model parameters and losses
MLFLOW_URI = "https://mlflow.lewagon.ai/"

# Dictionary containing paths to VM_folders for training, validation and test data
VM_path_dict = {
    'path_x': "../../raw_data/train_RGB_tiles_jpeg/",
    'path_y': "../../raw_data/train_mask_tiles_jpeg/"
}
# ~~~~~~~~~~~~~ testing only
test_pred_root_path = 'lwb_smr/data/demo_files_2/'
# ~~~~~~~~~~~~~

pred_root_path = 'lwb_smr/data/demo_files/'
prediction_path_dict = {
    'all_files_here': pred_root_path, # NOT!: '../lwb_smr/data/demo_files/'
    # N_OTE BENE this is used from lwb_smr_app => please DO NOT alter
    # USE 'all_files_here' instead (hilariouly named) 'input_image': pred_root_path+"prediction/google_map_images/",
    'model_path': pred_root_path+"prediction/",
    'output_tiles_path': pred_root_path+"prediction/raw_image_tiles/",
    'prediction_output_images_path': pred_root_path+"prediction/predicted_tiles_output/",
}



# SET NAMES OF 3 .csv FILES
csv_path_dict = {
    'train_csv': "image_datasets_csv/train_dataset.csv",
    'val_csv': "image_datasets_csv/validation_dataset.csv",
    'test_csv': "image_datasets_csv/test_dataset.csv"
}


predict_paths_dict = {
    'input_image': "/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/",
    'output_tiles_path': "/home/joshk/code/lwb_smr/lwb_smr/raw_data/prediction/raw_image_tiles/",
    'prediction_output_images_path': "/home/joshk/code/lwb_smr/lwb_smr/raw_data/prediction/predicted_tiles_output/",
    'model_path': "/home/joshk/code/lwb_smr/lwb_smr/raw_data/prediction/",
}
