from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------
# Base Model Evaluator Class
# ---------------------------------------------------------
class BaseModelEvaluator(ABC):
    """
    Abstract Base Class for Model Evaluators.
    All evaluators must implement the "evaluate" method.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def evaluate(self, **kwargs) -> dict:
        """
        Evaluate the model by calculating the generalization performance.

        Returns:
            dict: A dictionary containing the evaluation results.
        """
        pass


# ---------------------------------------------------------
# Generalization Model Evaluator Class
# ---------------------------------------------------------
class GeneralizationModelEvaluator(BaseModelEvaluator):
    """
    Evaluate the model by calculating the generalization performance.

    Attributes:
        top_k (int): The number of top items to recommend. (default: 10)
        sample_n (int): The number of users to sample for evaluation. (default: 1000)
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, top_k: int = 10, sample_n: int = 1000, random_state: int = None, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.sample_n = sample_n
        self.random_state = random_state

    def evaluate(self, df_test: pd.DataFrame, df_train: pd.DataFrame, user_factors: np.ndarray, product_factors: np.ndarray) -> dict:
        """
        Evaluate the model by calculating the generalization performance using the user and product factors.

        Args:
            df_test (pd.DataFrame): The dataframe containing the test data.
            df_train (pd.DataFrame): The dataframe containing the train data.
            user_factors (np.ndarray): The user factors. (shape: (num_users, vector_dimension))
            product_factors (np.ndarray): The product factors. (shape: (num_products, vector_dimension))

        Returns:
            dict: A dictionary containing the generalization performance evaluation results.
            - HitRate : The proportion of users for whom at least one relevant item is recommended. (1 if hit, else 0)
            - NDCG : A measure of ranking quality that considers the position of relevant items. (Higher is better)
            - Precision : The proportion of recommended items that are relevant. (TP / K)
            - Recall : The proportion of relevant items that are successfully recommended. (TP / Total Relevant)
        """
        print(f"⚙️ Evaluating Generalization Performance (Top-K : {self.top_k}, Sample N : {self.sample_n})...")

        # Test Set : Items the user actually interacted with in the future
        test_items_dict = df_test.groupby("user_idx")["product_idx"].apply(set).to_dict()

        # Train Set : Items the user has already interacted with (Exclude from recommendation)
        train_items_dict = df_train.groupby("user_idx")["product_idx"].apply(set).to_dict()

        # Sample Users for Evaluation
        test_users = np.array(list(test_items_dict.keys()))

        if len(test_users) > self.sample_n:
            np.random.seed(self.random_state)
            sampled_users = np.random.choice(test_users, self.sample_n, replace=False)
        else:
            sampled_users = test_users

        # Main Evaluation Loop
        metrics = {"HitRate": [], "NDCG": [], "Precision": [], "Recall": []}

        for idx, u_idx in enumerate(sampled_users):
            # Progress Log (Every 100 users)
            if (idx + 1) % 100 == 0:
                print(f"    ... Processing {idx + 1}/{len(sampled_users)} Users")

            # (1) Retrieve Ground Truth & Seen Items
            actual_items = test_items_dict[u_idx]  # Ground Truth
            seen_items = train_items_dict.get(u_idx, set())  # Already Seen Items

            # (2) Calculate Prediction Scores (Dot Product)
            # SVD Model: User Vector * Product Matrix Transpose
            u_vector = user_factors[u_idx]
            scores = np.dot(product_factors, u_vector)

            # (3) Mask Already Seen Items (Assign -infinity to their scores)
            items_to_mask = list(seen_items - actual_items)
            if len(items_to_mask) > 0:
                scores[items_to_mask] = -np.inf

            # (4) Extract Top-K Items (Using Fast Sort)
            top_k_indices = np.argpartition(scores, -self.top_k)[-self.top_k :]

            # Sort by score in descending order
            top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]

            # (5) Calculate Metrics
            num_correct = 0
            dcg = 0.0
            idcg = 0.0

            for i, item_idx in enumerate(top_k_indices):
                if item_idx in actual_items:
                    num_correct += 1
                    # Calculate DCG (Numerator of NDCG)
                    # Use log2(i + 2) because the rank starts from 1
                    dcg += 1.0 / np.log2(i + 2)

            # Calculate IDCG (Ideal DCG) - The best possible score
            num_actual = len(actual_items)
            for i in range(min(num_actual, self.top_k)):
                idcg += 1.0 / np.log2(i + 2)

            # Append results to metrics list
            metrics["HitRate"].append(1 if num_correct > 0 else 0)
            metrics["NDCG"].append(dcg / idcg if idcg > 0 else 0)
            metrics["Precision"].append(num_correct / self.top_k)
            metrics["Recall"].append(num_correct / num_actual)

        # Calculate Final Results
        results = {k: np.mean(v) for k, v in metrics.items()}

        return results


# ---------------------------------------------------------
# Engineering Model Evaluator Class
# ---------------------------------------------------------
class EngineeringModelEvaluator(BaseModelEvaluator):
    """
    Evaluate the model by calculating the engineering performance.
    Focuses on Reconstruction Error (RMSE) and Explained Variance Ratio (Explained Variance Ratio).
    This is primarily used for model selection and hyperparameter tuning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(
        self,
        df_eval: pd.DataFrame,
        svd_model: Any = None,
        user_factors: np.ndarray = None,
        product_factors: np.ndarray = None,
        target_col: str = "raw_weight",
    ) -> dict:
        """
        Evaluate the engineering performance by calculating the reconstruction error (RMSE) and explained variance ratio (Explained Variance Ratio).

        Args:
            df_eval (pd.DataFrame): The dataframe containing actual weights for RMSE calculation.
            svd_model (Any): The trained model object to access attribute like "explained_variance_ratio_".
            user_factors (np.ndarray): The user factors. (shape: (num_users, vector_dimension))
            product_factors (np.ndarray): The product factors. (shape: (num_products, vector_dimension))
            target_col (str): The column name of the target variable. (default: "raw_weight")

        Returns:
            dict: A dictionary containing the engineering performance evaluation results.
            - Explained Variance Ratio : The proportion of variance in the target variable that is explained by the model.
            - RMSE : The root mean squared error of the reconstruction.
        """
        print(f"⚙️ Evaluating Engineering Performance...")

        metrics = {}

        # (1) Explained Variance Ratio
        if svd_model is not None:
            explained_variance_ratio = svd_model.explained_variance_ratio_.sum()
            metrics["EVR"] = explained_variance_ratio

        # (2) Training RMSE
        if user_factors is not None and product_factors is not None:
            # Optimized Data Extraction (Vectorized)
            u_idx = df_eval["user_idx"].values
            p_idx = df_eval["product_idx"].values
            actual_weights = df_eval[target_col].values

            # Safety Check : Ensure Indices are within bounds
            valid_mask = (u_idx < user_factors.shape[0]) & (p_idx < product_factors.shape[0])

            if not valid_mask.all():
                out_count = len(valid_mask) - valid_mask.sum()
                print(f"⚠️ Warning : Found {out_count} out-of-bounds indices detected in training data.")

                # Filter out invalid indices
                u_idx = u_idx[valid_mask]
                p_idx = p_idx[valid_mask]
                actual_weights = actual_weights[valid_mask]

            # Calculate Prediction Scores (Dot Product)
            u_vectors = user_factors[u_idx]
            p_vectors = product_factors[p_idx]

            # Vectorized Calculation (Sum of Products)
            scores = np.sum(u_vectors * p_vectors, axis=1)

            # Calculate RMSE
            mse = mean_squared_error(actual_weights, scores)
            metrics["RMSE"] = np.sqrt(mse)

        return metrics
