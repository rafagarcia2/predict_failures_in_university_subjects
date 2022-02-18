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
        sep=";"
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
        sep=";"
    )

    logger.info("Importated Files!")
    matriculas.media_final = matriculas.media_final.apply(lambda x: float(x.replace(",", ".")))
    matriculas.nota = matriculas.nota.apply(lambda x: float(x.replace(",", ".")))

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

    turmas.ano_periodo = turmas.ano_periodo.apply(lambda x: int(x.split("_")[1] + x[-1]))

    lista_semestres_analisados = [20181, 20182, 20191, 20192]

    tx_reprovacao_do_professor_por_turma = {}
    for semestre in lista_semestres_analisados:
        turmas_anteriores = turmas.query(f"ano_periodo < {semestre}")
        
        turmas_atuais = turmas.query(f"ano_periodo == {semestre}")
        for index, row in turmas_atuais.iterrows():
            turmas_do_professor = turmas_anteriores[turmas_anteriores.id_docente_interno == row.id_docente_interno]
            if turmas_do_professor.shape[0] > 0:
                tx_reprovacao_do_professor_por_turma[row['id_turma']] = turmas_do_professor.tx_reprovacao.mean()
    
    turmas['professor_tx_reprovao'] = turmas.id_turma.apply(lambda x: tx_reprovacao_do_professor_por_turma[x] if x in tx_reprovacao_do_professor_por_turma.keys() else -1)

    ## Feature 2: desempenho_exatas
    id_turmas_c2 = turmas[turmas.id_componente_curricular.isin(['48584', 57588, '2051052'])].id_turma.unique()

    matriculas_c2 = matriculas[matriculas.id_turma.isin(id_turmas_c2)]
    matriculas_anteriores = matriculas[~matriculas.id_turma.isin(id_turmas_c2)]

    matriculas_c2 = matriculas_c2.query(f"unidade == 1")

    matriculas_c2['desempenho_exatas'] = -1
    tx_reprovacao_do_professor_por_turma = {}
    for semestre in lista_semestres_analisados:
        turmas_anteriores = turmas.query(f"ano_periodo < {semestre}")
        matr_antigas = matriculas_anteriores[matriculas_anteriores.id_turma.isin(turmas_anteriores.id_turma.unique())]
        
        turmas_atuais = turmas.query(f"ano_periodo == {semestre}")
        matricula_atual = matriculas_c2[matriculas_c2.id_turma.isin(turmas_atuais.id_turma.unique())]
        for index, row in matricula_atual.iterrows():
            aluno_matr_ant = matr_antigas[matr_antigas.discente == row['discente']]
            
            
            if aluno_matr_ant.shape[0] > 0:
                matriculas_c2.at[index, 'desempenho_exatas'] = round(aluno_matr_ant.media_final.mean(), 2)
            else:
                matriculas_c2.at[index, 'desempenho_exatas'] = -1

    media_desempenho_exatas = matriculas_c2[matriculas_c2.desempenho_exatas != -1].desempenho_exatas.mean()
    matriculas_c2.desempenho_exatas = matriculas_c2.desempenho_exatas.apply(lambda x: media_desempenho_exatas if x == -1 else x)

    ## Feature 3: historico_reprovacao
    matriculas_c2['historico_reprovacao'] = -1

    tx_reprovacao_do_professor_por_turma = {}
    for semestre in lista_semestres_analisados:
        turmas_anteriores = turmas.query(f"ano_periodo < {semestre}")
        matr_antigas = matriculas[matriculas.id_turma.isin(turmas_anteriores.id_turma.unique())]
        
        turmas_atuais = turmas.query(f"ano_periodo == {semestre}")
        matricula_atual = matriculas_c2[matriculas_c2.id_turma.isin(turmas_atuais.id_turma.unique())]
        for index, row in matricula_atual.iterrows():
            aluno_matr_ant = matr_antigas[matr_antigas.discente == row['discente']]
            if aluno_matr_ant.shape[0] > 0:
                matriculas_c2.at[index, 'historico_reprovacao'] = sum(aluno_matr_ant.reprovou) / aluno_matr_ant.shape[0]
    
    ## Feature 4: primeira_vez_pagando
    matriculas_c2['primeira_vez_pagando'] = 0

    tx_reprovacao_do_professor_por_turma = {}
    for semestre in lista_semestres_analisados:
        turmas_anteriores = turmas.query(f"ano_periodo < {semestre}")
        matr_antigas = matriculas_c2[matriculas_c2.id_turma.isin(turmas_anteriores.id_turma.unique())]
        
        turmas_atuais = turmas.query(f"ano_periodo == {semestre}")
        matricula_atual = matriculas_c2[matriculas_c2.id_turma.isin(turmas_atuais.id_turma.unique())]
        for index, row in matricula_atual.iterrows():
            aluno_matr_ant = matr_antigas[matr_antigas.discente == row['discente']]
            
            matriculas_c2.at[index, 'primeira_vez_pagando'] = aluno_matr_ant.shape[0]
    
    ## Feature 5: desvio_padra_para_turma

    def nota_em_relacao_a_turma(nota, id_turma):
        media_turma = statisticas_turmas['media_final']['mean'][id_turma]
        desvio_padrao = statisticas_turmas['media_final']['std'][id_turma]
        
        if desvio_padrao > 0:
            return (nota - media_turma) / desvio_padrao
        return (nota - media_turma)

    matriculas_c2['n1_std_turma'] = matriculas_c2.apply(lambda x: nota_em_relacao_a_turma(x['nota'], x['id_turma']), axis=1)

    ## Feature 6: nota_unidade1

    ## Feature 7: professor_c1
    matriculas_c2['prof_c1_tx_reprovacao'] = -1

    tx_reprovacao_do_professor_por_turma = {}
    for semestre in lista_semestres_analisados:
        turmas_anteriores = turmas.query(f"ano_periodo < {semestre}")
        matr_antigas = matriculas[matriculas.id_turma.isin(turmas_anteriores.id_turma.unique())]
        
        turmas_atuais = turmas.query(f"ano_periodo == {semestre}")
        matricula_atual = matriculas_c2[matriculas_c2.id_turma.isin(turmas_atuais.id_turma.unique())]
        for index, row in matricula_atual.iterrows():
            aluno_matr_ant = matr_antigas[matr_antigas.discente == row['discente']]
            aluno_turmas_ant = turmas_anteriores[turmas_anteriores.id_turma.isin(aluno_matr_ant.id_turma.unique())]
            aluno_turmas_ant = aluno_turmas_ant[aluno_turmas_ant.id_componente_curricular.isin(['48582', '57587', '2050801'])]
            if aluno_turmas_ant.shape[0] > 0:
                matriculas_c2.at[index, 'prof_c1_tx_reprovacao'] = aluno_turmas_ant.professor_tx_reprovao.mean()

    media_prof_c1_tx_reprovacao = matriculas_c2[~matriculas_c2.prof_c1_tx_reprovacao.isna()].prof_c1_tx_reprovacao.mean()

    matriculas_c2.prof_c1_tx_reprovacao.fillna(media_prof_c1_tx_reprovacao, inplace=True)

    ## Adionando a dificuldade do professor

    matriculas_c2 = pd.merge(matriculas_c2, turmas[['id_turma', 'professor_tx_reprovao']], on='id_turma', how='left')

    matriculas_c2 = matriculas_c2.drop_duplicates()
    matriculas_c2.professor_tx_reprovao.fillna(matriculas_c2.professor_tx_reprovao.median(), inplace=True)

    logger.info("Creating artifact")

    matriculas_c2.to_csv(args.artifact_name)

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(args.artifact_name)

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
