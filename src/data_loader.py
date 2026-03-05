import sqlite3
import pandas as pd

def list_tables(db_path: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
            conn
        )["name"].tolist()
    return tables

def load_data(db_path: str) -> pd.DataFrame:
    tables = list_tables(db_path)
    if not tables:
        raise ValueError(f"No tables found in database: {db_path}")

    table = tables[0]
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table};", conn)

    return df
