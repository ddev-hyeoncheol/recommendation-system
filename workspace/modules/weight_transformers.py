from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Base Weight Transformer Class
# ---------------------------------------------------------
class BaseWeightTransformer(ABC):
    """
    Abstract Base Class for Weight Transformers.
    All transformers must implement the "transform" method.

    Attributes:
        raw_weight_col (str): The name of the column containing the original weights. (default: "raw_weight")
    """

    def __init__(self, raw_weight_col: str = "raw_weight", **kwargs):
        self.kwargs = kwargs
        self.raw_weight_col = raw_weight_col

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """
        Transform the raw weights into new weights based on the provided weight strategy.
        It expects the input "df" to be a pandas DataFrame containing the raw weights.

        Args:
            df (pd.DataFrame): The input dataframe containing the original weights.

        Returns:
            pd.Series: The series with the transformed weights.
        """
        pass


# ---------------------------------------------------------
# Log Normalization Weight Transformer Class
# ---------------------------------------------------------
class LogNormalizationWeightTransformer(BaseWeightTransformer):
    """
    Transform the raw weights into log normalized weights using log(1 + x).
    This reduces the impact of outliers and compresses the weight distribution.

    Attributes:
        base (float): The base of the logarithm. (default: np.e, natural log)
    """

    def __init__(self, base: float = np.e, **kwargs):
        super().__init__(**kwargs)
        self.base = base

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """
        Transform the raw weights into log normalized weights.

        Args:
            df (pd.DataFrame): The input dataframe containing the original weights.

        Returns:
            pd.Series: The series with the log-normalized weights.
        """
        print(f"⚙️ Applying Log Normalization (Base : {self.base:.2f})...")

        raw_weights = df[self.raw_weight_col].astype(np.float32)

        # Apply log(1 + x) transformation
        # Adding 1 ensures log(0) is avoided and log(1) = 0
        if self.base == np.e:
            weights = np.log1p(raw_weights)
        else:
            weights = np.log1p(raw_weights) / np.log(self.base)

        return weights


# ---------------------------------------------------------
# BM25 Weight Transformer Class
# ---------------------------------------------------------
class BM25WeightTransformer(BaseWeightTransformer):
    """
    Apply BM25 transformation to the dataframe to address popularity bias and normalize user activity levels.

    The formula used is:
        IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (L_u / L_avg)))

    Attributes:
        user_col (str): The name of the column containing the user IDs. (default: "user_idx")
        item_col (str): The name of the column containing the item IDs. (default: "product_idx")
        raw_weight_col (str): The name of the column containing the original weights. (default: "raw_weight")
        k1 (float): Saturation parameter. Controls how quickly the weight saturates. (default: 1.2)
        b (float): Length normalization parameter. Controls how much to penalize heavy users. (default: 0.75)
    """

    def __init__(
        self,
        user_col: str = "user_idx",
        item_col: str = "product_idx",
        raw_weight_col: str = "raw_weight",
        k1: float = 1.2,
        b: float = 0.75,
        **kwargs,
    ):
        super().__init__(raw_weight_col=raw_weight_col, **kwargs)
        self.user_col = user_col
        self.item_col = item_col
        self.k1 = k1
        self.b = b

    def transform(self, df: pd.DataFrame, avg_doc_length: float = None, doc_freq_col: str = None, num_docs: int = None, **kwargs) -> pd.Series:
        """
        Transform the raw weights into BM25-weighted weights.

        Args:
            df (pd.DataFrame): The input dataframe containing the original weights.
            avg_doc_length (float): The average document length. If None, calculated from df. (default: None)
            doc_freq_col (str): The column name containing the document frequencies. If None, calculated from df. (default: None)
            num_docs (int): The total number of documents (users). If None, calculated from df. (default: None)

        Returns:
            pd.Series: The series with the BM25-transformed weights.
        """
        print(f"⚙️ Applying BM25 Transformation (K1 : {self.k1:.2f}, B : {self.b:.2f})...")

        raw_weights = df[self.raw_weight_col].astype(np.float32)

        # Calculate Statistics
        # N : Total number of Users (Documents)
        if num_docs is None:
            N = df[self.user_col].nunique()
        else:
            N = num_docs

        # n_i : Document Frequency per Product (Number of Interactions for each Product)
        if doc_freq_col is None:
            n_i_counts = df.groupby(self.item_col)[self.user_col].count()
            n_i_series = df[self.item_col].map(n_i_counts).fillna(1)
        else:
            n_i_series = df[doc_freq_col].fillna(1)

        # L_u : User Activity Length (Sum of Raw Weights for each User)
        l_u_counts = df.groupby(self.user_col)[self.raw_weight_col].sum()
        l_u_series = df[self.user_col].map(l_u_counts)

        # L_avg : Average User Activity Length (Mean of User Activity Lengths)
        if avg_doc_length is None:
            l_avg = l_u_counts.mean()
        else:
            l_avg = avg_doc_length

        # Calculate IDF (Inverse Document Frequency)
        # IDF = log((N - n_i + 0.5) / (n_i + 0.5) + 1)
        idf = np.log((N - n_i_series + 0.5) / (n_i_series + 0.5) + 1)

        # Calculate BM25 Weight
        # BM25 = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (L_u / L_avg)))
        numerator = raw_weights * (self.k1 + 1)
        denominator = raw_weights + self.k1 * (1 - self.b + self.b * (l_u_series / l_avg))

        # Assign Weight
        weights = idf * numerator / denominator

        return weights
