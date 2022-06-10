"""
File to store parameters for reuse

"""

# MLFlow URI to save model parameters and losses
MLFLOW_URI = "https://mlflow.lewagon.ai/"


# Dictionary containing paths to VM_folders for training, validation and test data
VM_path_dict = {
    'path_x': '../../raw_data/train_RGB_tiles_jpeg/',
    'path_y': '../../raw_data/train_mask_tiles_jpeg/'
}


# SET NAMES OF 3 .csv FILES

csv_path_dict = {
    'train_csv': "/home/joshk/code/lwb_smr/lwb_smr/raw_data/image_datasets_csv/train_dataset.csv",
    'val_csv': "/home/joshk/code/lwb_smr/lwb_smr/raw_data/image_datasets_csv/validation_dataset.csv",
    'test_csv': "/home/joshk/code/lwb_smr/lwb_smr/raw_data/image_datasets_csv/test_dataset.csv"
}
