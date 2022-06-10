
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

from lwb_smr.params import MLFLOW_URI

class PushMLFlow():
    '''
        MLFLOW_URI = "https://mlflow.lewagon.ai/"
        EXPERIMENT_NAME = "[UK] [LONDON] [SOLAR_ROOF] TEST RUN" # template
        EXPERIMENT_TAGS = {
            'USER': 'test_user',
            'RUN NAME': 'test runs',
            'VERSION': '1.0.1',
            'DESCRIPTION': 'testing MLFlow Pipeline. Model - basic U-Net structure, 2 epochs, 15 images'
        }
    '''

    def __init__(self, experiment_name, experiment_tags):
        self.experiment_name = experiment_name
        self.experiment_tag = experiment_tags

        pass
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id, tags=self.experiment_tag)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
