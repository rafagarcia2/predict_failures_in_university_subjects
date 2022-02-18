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
from omegaconf import DictConfig, OmegaConf

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

    _ = mlflow.run(
        os.path.join(root_path, "preprocessing"),
        "main",
        parameters={
            "matr_input_artifact": "matriculas.csv:latest",
            "turmas_input_artifact": "turmas.csv:latest",
            "artifact_name": "preprocessed_data.csv",
            "artifact_type": "processed_data",
            "artifact_description": "Cleaned data"
        },
    )

    _ = mlflow.run(
        os.path.join(root_path, "segregation"),
        "main",
        parameters={
            "input_artifact": "preprocessed_data.csv:latest",
            "artifact_root": "data",
            "artifact_type": "segregated_data",
            "test_size": config["data"]["test_size"],
            "stratify": config["data"]["stratify"],
            "random_state": config["main"]["random_seed"]
        }
    )

    # Serialize random forest configuration
    model_config = os.path.abspath("random_forest_config.yml")

    with open(model_config, "w+") as fp:
        fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

    _ = mlflow.run(
        os.path.join(root_path, "random_forest"),
        "main",
        parameters={
            "train_data": "train_data.csv:latest",
            "model_config": model_config,
            "export_artifact": config["random_forest_pipeline"]["export_artifact"],
            "random_seed": config["main"]["random_seed"],
            "val_size": config["data"]["val_size"],
            "stratify": config["data"]["stratify"]
        }
    )

    _ = mlflow.run(
        os.path.join(root_path, "evaluate"),
        "main",
        parameters={
            "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
            "test_data": "test_data.csv:latest"
        }
    )

if __name__ == "__main__":
    process_args()
