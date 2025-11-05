from typing import List
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, KBinsDiscretizer

def build_transformer_for_tree(categorical_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('income_log', FunctionTransformer(np.log1p), ['person_income']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],    
        remainder='passthrough'
        )


def build_transformer_for_regression(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('income_log', FunctionTransformer(np.log1p), ['person_income']),
            ('age_binned', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'), ['person_age']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],    
        remainder='drop'
    )