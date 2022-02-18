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

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = list(config["main"]["execute_steps"])

    # Download step
    if "download" in steps_to_execute:
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

    if "preprocess" in steps_to_execute:
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

    if "data_checks" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "data_checks"),
            "main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": "preprocessed_data.csv:latest",
                "ks_alpha": config["data"]["ks_alpha"]
            }
        )

    if "segregate" in steps_to_execute:
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

    if "random_forest" in steps_to_execute:
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

    if "evaluate" in steps_to_execute:
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
