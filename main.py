"""
Creator: Rafael Garcia
Date: 16 Fev. 2022
Predict Failures in the University Subjects (UFRN)
ML Pipeline Components in MLflow
In this module we will build a MLflow component.
"""
import mlflow
import os
import wandb
import hydra
from omegaconf import DictConfig

# This automatically reads in the configuration
@hydra.main(config_name='config')
def process_args(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        os.path.join(root_path, "download"),
        "main",
        parameters={
            "file_url": config["data"]["enrollments_url"],
            "artifact_name": "matriculas.csv",
            "artifact_type": "raw_data",
            "artifact_description": "enrollments in subjects"
        },
    )

    _ = mlflow.run(
        os.path.join(root_path, "download"),
        "main",
        parameters={
            "file_url": config["data"]["classes_url"],
            "artifact_name": "turmas.csv",
            "artifact_type": "raw_data",
            "artifact_description": "classes"
        },
    )
if __name__ == "__main__":
    process_args()
