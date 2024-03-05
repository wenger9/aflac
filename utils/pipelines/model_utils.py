from dataclasses import dataclass
import json
from typing import Union

import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.tree._tree import Tree
from sklearn.ensemble._forest import BaseForest


class GetDummies(sklearn.base.TransformerMixin):
    """Fast one-hot-encoder that makes use of pandas.get_dummies() safely
    on train/test splits.
    """

    def __init__(self, dtypes=None):
        self.input_columns = None
        self.final_columns = None
        if dtypes is None:
            dtypes = [object, "category"]
        self.dtypes = dtypes

    def fit(self, X, y=None, **kwargs):
        """
        Fit model method
        """
        self.input_columns = list(X.select_dtypes(self.dtypes).columns)
        X = pd.get_dummies(X, columns=self.input_columns)
        self.final_columns = X.columns
        return self

    def transform(self, X, y=None, **kwargs):
        """
        Model transform method
        """
        X = pd.get_dummies(X, columns=self.input_columns)
        feature_columns = X.columns
        # if columns in X had values not in the data set used during
        # fit add them and set to 0
        missing = set(self.final_columns) - set(feature_columns)
        for column in missing:
            X[column] = 0
        # remove any new columns that may have resulted from values in
        # X that were not in the data set when fit
        return X[self.final_columns]

    def get_feature_names(self):
        """
        Get tuple with final column names
        """
        return tuple(self.final_columns)


@dataclass
class ModelConfig:
    """Model target and features."""
    target_col: str
    pos_target: Union[str, int]
    feature_cols: list
    cat_cols: list
    num_cols: list

    @classmethod
    def load(cls, model_name: str) -> "ModelConfig":
        """
        Get model configuration via mlflow
        """
        model_config = mlflow.artifacts.load_dict(f"model_configs/{model_name}.json")
        model_config = cls.from_dict(model_config)
        return model_config

    @classmethod
    def from_dict(cls, model_config: dict) -> "ModelConfig":
        """
        Get model configuration from dict
        """
        model_config = cls(
            target_col=model_config["target"],
            pos_target=model_config["positive_target"],
            feature_cols=list(model_config["feature_types"].keys()),
            cat_cols=[f for f, t in model_config["feature_types"].items() if t == "categorical"],
            num_cols=[f for f, t in model_config["feature_types"].items() if t == "numeric"]
        )
        return model_config


def tree_nbytes(tree: Tree) -> int:
    """Get size in bytes of arrays in a decision tree."""
    size = 0
    for attr in dir(tree):
        obj = getattr(tree, attr)
        if isinstance(obj, np.ndarray):
            size += obj.nbytes
    return size


def get_estimator_size(pipeline: Pipeline) -> str:
    """
    Get the model size. It is useful for debug
    """
    estimator = pipeline.steps[-1][-1]
    if isinstance(estimator, BaseForest):
        nbytes = sum(tree_nbytes(t.tree_) for t in estimator.estimators_)
        nodes_count = sum(t.tree_.node_count for t in estimator.estimators_)
    else:
        nbytes = None
        nodes_count = None

    stats = json.dumps(
        {
            "nbytes": nbytes,
            "nodes_count": nodes_count,
        }
    )

    return stats
