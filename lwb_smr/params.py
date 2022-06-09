"""
File to store parameters for reuse

"""

# MLFlow URI to save model parameters and losses
MLFLOW_URI = "https://mlflow.lewagon.ai/"


# Dictionary containing paths to VM_folders for training, validation and test data
VM_path_dict = {
    'path_X_train': '../raw_data/AerialImageDataset/train/images/',
    'path_y_train': '../raw_data/AerialImageDataset/train/gt/',
    'path_X_val': '',
    'path_y_val': '',
    'path_X_test': '',
    'path_y_test': ''
}
