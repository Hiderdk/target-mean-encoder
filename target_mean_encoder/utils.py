import pandas as pd
from typing import List


def detect_categories_by_dtype(df: pd.DataFrame, targeted_type: str) -> List[str]:
    categorical_columns = []
    for column in df.columns:
        if df[column].dtype == targeted_type:
            categorical_columns.append(column)
    return categorical_columns


def detect_categories_by_distinct_count(df: pd.DataFrame, max_distinct_count: int) -> List[str]:
    categorical_columns = []
    for column in df.columns:
        distinct_values_count = len(df[column].unique())
        if distinct_values_count <= max_distinct_count:
            categorical_columns.append(column
                                       )
    return categorical_columns
