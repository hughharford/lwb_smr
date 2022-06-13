# NOTE: mlflow original class is below, new class now in utils_class
#
# and others...
import tensorflow as tf
import tensorflow_addons as tfa

# need to specify the DICE LOSS to enable reloading the model
def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


def load_model_from_local_path(model_load_string_inc_path_n_h5):
    custom_objects_dict = {
                    'dice_loss': dice_loss
    }
    loaded_model = tf.keras.models.load_model(model_load_string_inc_path_n_h5, custom_objects=custom_objects_dict)

    return loaded_model


def aug_flip_l_r(image, mask):
  """
    Take one image and its mask and apply left-right flip transformation
    Return image and mask
    Applies function with a 30% chance of transformation occuring
    """
  choice = tf.random.uniform((), minval=0, maxval=1)
  if True and choice < 0.3:
    image = tf.image.flip_left_right(image)
    mask = tf.image.flip_left_right(mask)
    return image, mask
  return image, mask


def aug_flip_u_d(image, mask):
  """
    Take one image and its mask and apply up-down flip transformation
    Return image and mask
    Applies function with a 30% chance of transformation occuring
    """
  choice = tf.random.uniform((), minval=0, maxval=1)
  if True and choice < 0.3:
    image = tf.image.flip_up_down(image)
    mask = tf.image.flip_up_down(mask)
    return image, mask
  return image, mask


def aug_rotate(image, mask):
  """
    Take one image and its mask and apply rotation transformation
    Return image and mask
    Applies function with a 30% chance of transformation occuring
    """
  choice = tf.random.uniform((), minval=0, maxval=1)
  if True and choice < 0.3:
    angle = tf.random.uniform((), minval=-0.5, maxval=0.5) # apporox pi/6 rads = 30degrees
    image = tfa.image.rotate(image, angles=angle)
    mask = tfa.image.rotate(mask, angles=angle)

    # crop image to ensure that edges arent blank
    # not required (yannis)
    # image = tf.image.central_crop(image,0.8)
    # mask = tf.image.central_crop(mask,0.8)


    return image, mask
  return image, mask














# import mlflow
# from mlflow.tracking import MlflowClient
# from memoized_property import memoized_property

# MLFLOW_URI = "https://mlflow.lewagon.ai/"

# class PushMLFlow():
#     '''
#         MLFLOW_URI = "https://mlflow.lewagon.ai/"
#         EXPERIMENT_NAME = "[UK] [LONDON] [SOLAR_ROOF] TEST RUN" # template
#         EXPERIMENT_TAGS = {
#             'USER': 'test_user',
#             'RUN NAME': 'test runs',
#             'VERSION': '1.0.1',
#             'DESCRIPTION': 'testing MLFlow Pipeline. Model - basic U-Net structure, 2 epochs, 15 images'
#         }
#     '''

#     def __init__(self, experiment_name, experiment_tags):
#         self.experiment_name = experiment_name
#         self.experiment_tag = experiment_tags

#         pass
#     @memoized_property
#     def mlflow_client(self):
#         mlflow.set_tracking_uri(MLFLOW_URI)
#         return MlflowClient()

#     @memoized_property
#     def mlflow_experiment_id(self):
#         try:
#             return self.mlflow_client.create_experiment(self.experiment_name)
#         except BaseException:
#             return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

#     @memoized_property
#     def mlflow_run(self):
#         return self.mlflow_client.create_run(self.mlflow_experiment_id, tags=self.experiment_tag)

#     def mlflow_log_param(self, key, value):
#         self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

#     def mlflow_log_metric(self, key, value):
#         self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
