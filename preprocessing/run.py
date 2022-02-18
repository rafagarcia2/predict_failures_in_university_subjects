"""
Creator: Rafael Garcia
Date: 16 Fev. 2022
Predict Failures in the University Subjects (UFRN)
ML Pipeline Components in MLflow
In this module we will build a MLflow component.
"""
import argparse
import logging
import seaborn as sns
import pandas as pd
import numpy as np
import wandb

# configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
)

# reference for a logging obj
logger = logging.getLogger()


def process_args(args):

    run = wandb.init(job_type="preprocessing")

    logger.info("Downloading artifact - " + str(args.matr_input_artifact))
    matr_artifact = run.use_artifact("matriculas.csv:latest")
    matr_artifact_path = matr_artifact.file()

    turmas_artifact = run.use_artifact("turmas.csv:latest")
    turmas_artifact_path = turmas_artifact.file()

    matriculas = pd.read_csv(
        matr_artifact_path,
        skiprows=1,
        names=[
            "discente",
            "faltas_unidade",
            "id_turma",
            "media_final",
            "nota",
            "numero_total_faltas",
            "reposicao",
            "unidade",
            "reprovou",
        ],
    )

    turmas = pd.read_csv(
        turmas_artifact_path,
        skiprows=1,
        names=[
            "id_turma",
            "id_componente_curricular",
            "id_docente_interno",
            "ano_periodo",
        ],
    )

    logger.info("Importated Files!")
    matriculas.media_final = matriculas.media_final.apply(
        # lambda x: float(x.replace(",", ".")) if isinstance(x, str) else x
        lambda x: x if isinstance(x, float) else np.nan
    )

    matriculas = matriculas[~matriculas.media_final.isna()]

    statisticas_turmas = matriculas.groupby("id_turma").agg(
        {"reprovou": ["sum", "count"], "media_final": ["mean", "std"]}
    )

    ## Feature 1: dificuldade_professor

    def get_dificuldade_prod(row):
        stats_reprovou = statisticas_turmas["reprovou"]
        if stats_reprovou["sum"].get(row) and stats_reprovou["count"].get(row):
            return stats_reprovou["sum"].get(row) / stats_reprovou["count"].get(row)
        else:
            return np.nan

    turmas["tx_reprovacao"] = turmas.id_turma.apply(get_dificuldade_prod)

    logger.info("Creating artifact")

    turmas[["id_turma", "tx_reprovacao"]].to_csv("clean_data.csv")

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file("clean_data.csv")

    logger.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to W&B",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--matr_input_artifact",
        type=str,
        help="Fully-qualified name for the matriculas input artifact",
        required=True,
    )

    parser.add_argument(
        "--turmas_input_artifact",
        type=str,
        help="Fully-qualified name for the turmas input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    ARGS = parser.parse_args()

    process_args(ARGS)


names=[
            "discente", "faltas_unidade", "id_turma", "media_final", "nota", "numero_total_faltas", "reposicao", "unidade", "reprovou",
        ],