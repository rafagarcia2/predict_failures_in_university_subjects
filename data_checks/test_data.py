import pandas as pd
import scipy.stats

# Determinstic Test
def test_column_presence_and_type(data):
    
    # Disregard the reference dataset
    _, df = data

    required_columns = {
        "discente": pd.api.types.is_object_dtype,
        "id_turma": pd.api.types.is_int64_dtype,
        "numero_total_faltas": pd.api.types.is_float_dtype,
        "desempenho_exatas": pd.api.types.is_float_dtype,
        "historico_reprovacao": pd.api.types.is_int64_dtype,
        "prof_c1_tx_reprovacao": pd.api.types.is_int64_dtype,
        "professor_tx_reprovao": pd.api.types.is_float_dtype,
        "reprovou": pd.api.types.is_bool_dtype
    }

    # Check column presence
    assert set(df.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(df[col_name]), f"Column {col_name} failed test {format_verification_funct}"

# Deterministic Test
def test_class_names(data):
    
    # Disregard the reference dataset
    _, df = data

    # Check that only the known classes are present
    known_classes = [0, 1]

    assert df["reprovou"].isin(known_classes).all()


# Deterministic Test
def test_column_ranges(data):
    
    # Disregard the reference dataset
    _, df = data

    ranges = {
        "nota": (0, 10),
        "media_final": (0, 10),
        "numero_total_faltas": (0, 200),
        "desempenho_exatas": (0, 10),
        "historico_reprovacao": (-1, 1),
        "prof_c1_tx_reprovacao": (-1, 1),
        "professor_tx_reprovao": (-1, 1),
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert df[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={df[col_name].min()} and max={df[col_name].max()}"
        )