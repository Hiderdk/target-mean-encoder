import pandas as pd
from typing import Dict, List, Literal
from sklearn.base import BaseEstimator
import numpy as np
from pandas.core.common import SettingWithCopyWarning
import warnings

from models import FeatureTargetMean, TargetMean
from utils import detect_categories_by_distinct_count, detect_categories_by_dtype

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class TargetMeanEncoder(BaseEstimator):

    def __init__(self, target_name: str = "TARGET", smoothing_weight: float = 0, max_sample_size_smoothing=10,
                 handle_unknown='value',
                 auto_detect_categories_method: Literal["None", "dType", "distinct_value_count"] = "dType",
                 max_distinct_value_count=20
                 ):
        self._categorical_features: List[str] = []
        self._feature_to_target_means: Dict[str, FeatureTargetMean] = {}
        self._target_name: str = target_name
        self._auto_detect_categories_method = auto_detect_categories_method
        self._min_distinct_value_count = max_distinct_value_count
        self._smoothing_parameter: float = smoothing_weight
        self._all_target_mean: float = 0
        self._max_sample_size_smoothing: int = max_sample_size_smoothing
        self._handle_unknown: str = handle_unknown
        self._all_column_names: List[str] = []
        self._transformed_categorical_feature_names: List[str] = []
        self._transformed_category_name_to_old_category_name_mapping: Dict[str, str] = {}

        if auto_detect_categories_method not in ["None", "dType", "distinct_value_count"]:
            raise ValueError(f"auto_detect_categories_method name {auto_detect_categories_method} is not valid")

    @property
    def transformed_category_name_to_old_category_name_mapping(self) -> Dict[str, str]:
        return self._transformed_category_name_to_old_category_name_mapping

    @property
    def all_column_names(self) -> List[str]:
        return self._all_column_names

    @property
    def transformed_categorical_feature_names(self) -> List[str]:
        return self._transformed_categorical_feature_names

    @property
    def feature_to_target_means(self) -> Dict[str, FeatureTargetMean]:
        return self._feature_to_target_means

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if len(X) == 0:
            raise ValueError("X has length 0")
        if len(y) == 0:
            raise ValueError("y has length 0")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")

        if not isinstance(y, pd.Series):
            raise TypeError("y is not a dataframe")

        self._categorical_features = self._get_categorical_feature_names(X)
        df = X
        df[self._target_name] = y

        for feature in self._categorical_features:
            target_mean = self._generate_target_means(df, feature)
            feature_to_target_mean = FeatureTargetMean(name=feature, target_means=target_mean,
                                                       if_unknown_mean=self._all_target_mean)
            self._feature_to_target_means[feature] = feature_to_target_mean

    def _get_categorical_feature_names(self, X: pd.DataFrame) -> List[str]:

        if self._auto_detect_categories_method == "None":
            return X.columns.tolist()

        elif self._auto_detect_categories_method == "dType":
            return detect_categories_by_dtype(X, "object")

        elif self._auto_detect_categories_method == "distinct_value_count":
            categorical_features_by_dtype = detect_categories_by_dtype(X, "object")
            categorical_features_by_count = detect_categories_by_distinct_count(X, self._min_distinct_value_count)
            return categorical_features_by_count + [
                c for c in categorical_features_by_dtype if c not in categorical_features_by_count]

    def _generate_target_means(self, df: pd.DataFrame, feature_name: str) -> Dict[str, TargetMean]:
        target_means: Dict[str, TargetMean] = {}
        distinct_values = df[feature_name].unique()
        self._all_target_mean = df[self._target_name].mean()

        for distinct_value_name in distinct_values:
            target_means[distinct_value_name] = self._calculate_target_mean(df, feature_name, distinct_value_name)

        return target_means

    def _calculate_target_mean(self, df: pd.DataFrame, feature_name: str, distinct_value_name: str) -> TargetMean:
        rows = df[df[feature_name] == distinct_value_name]
        if len(rows) == 0:
            return TargetMean(cell_name=distinct_value_name, mean=self._all_target_mean, observations=len(rows))

        sample_size_mean = rows[self._target_name].mean()
        base_weight = min(self._max_sample_size_smoothing, len(rows)) / self._max_sample_size_smoothing
        weighted_mean = sample_size_mean * base_weight + (1 - base_weight) * self._all_target_mean
        smoothed_mean = weighted_mean * (
                1 - self._smoothing_parameter) + self._smoothing_parameter * self._all_target_mean
        target_mean = TargetMean(distinct_value_name, smoothed_mean, base_weight=base_weight,
                                 weighted_mean=weighted_mean, observations=len(rows), sample_size_mean=sample_size_mean)

        return target_mean

    def transform(self, X: pd.DataFrame, overwrite_existing_column: bool = False, prefix: str = "target_mean",
                  suffix: str = "") -> pd.DataFrame:

        if not overwrite_existing_column and prefix == "" and suffix == "":
            raise ValueError("overwrite_existing_column cannot be set to false with no prefix and suffix name")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a dataframe")

        if len(self._feature_to_target_means) == 0:
            raise ValueError("Model has not not been fitted yet")

        self._all_column_names = X.columns.tolist()
        self._transformed_categorical_feature_names = []

        for feature_name in self._categorical_features:
            if feature_name not in X.columns:
                raise ValueError(f"cannot find categorical feature: {feature_name} in dataframe")

            X[feature_name] = self._transform_single_feature(X, feature_name=feature_name,
                                                             overwrite_existing_column=overwrite_existing_column,
                                                             prefix=prefix, suffix=suffix)

            return X

    def _transform_single_feature(self, X: pd.DataFrame, feature_name: str, overwrite_existing_column: bool,
                                  prefix: str, suffix: str):
        distinct_values = X[feature_name].unique()
        column_name = self._generate_new_column_name(feature_name, overwrite_existing_column, prefix, suffix)
        self._transformed_categorical_feature_names.append(column_name)
        self._transformed_category_name_to_old_category_name_mapping[column_name] = feature_name
        self._all_column_names.remove(feature_name)
        self._all_column_names.append(column_name)

        if column_name not in X.columns:
            X[column_name] = np.NAN
        for distinct_value_name in distinct_values:
            target_mean_value = self._calculate_target_mean_value(
                distinct_value_name, feature_name)

            X.loc[X[feature_name] == distinct_value_name, column_name] = target_mean_value

        if self._handle_unknown:
            X.loc[X[column_name].isna(), column_name] = self._feature_to_target_means[feature_name].if_unknown_mean

        return X[column_name]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, overwrite_existing_column: bool = False,
                      prefix: str = "target_mean", suffix: str = ""):
        self.fit(X, y)
        return self.transform(X, overwrite_existing_column, prefix, suffix)

    def _calculate_target_mean_value(self, distinct_value_name: str, feature_name: str) -> float:
        if distinct_value_name not in self._feature_to_target_means[feature_name].target_means:
            return self._feature_to_target_means[feature_name].if_unknown_mean

        return self._feature_to_target_means[feature_name].target_means[distinct_value_name].mean

    def _generate_new_column_name(self, feature_name: str, overwrite_existing_column: bool, prefix: str,
                                  suffix: str) -> str:
        if overwrite_existing_column:
            return feature_name

        new_feature_name = feature_name
        if prefix != "":
            new_feature_name = prefix + "_" + new_feature_name

        if suffix != "":
            new_feature_name = new_feature_name + "_" + suffix

        return new_feature_name
