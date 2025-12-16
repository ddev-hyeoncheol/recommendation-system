from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Base Data Splitter Class
# ---------------------------------------------------------
class BaseDataSplitter(ABC):
    """
    Abstract Base Class for Data Splitting Strategies.
    All splitters must implement the "split" method.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataframe into training and testing sets.

        Args:
            df (pd.DataFrame): The dataframe to split.

        Returns:
            tuple: A tuple containing the training and testing sets. (df_train, df_test)
        """
        pass


# ---------------------------------------------------------
# Stratified Data Splitter Class
# ---------------------------------------------------------
class StratifiedDataSplitter(BaseDataSplitter):
    """
    Split interaction data using a stratified strategy per user.

    This strategy ensures that:
    1. Users with fewer than "min_interactions" are forced into the training set. (Cold Start Prevention)
    2. Eligible users have their interactions split based on the "test_ratio".

    Attributes:
        test_ratio (float): The proportion of the dataset to include in the test split. (default: 0.2)
        min_interactions (int): Minimum number of interactions required to include a user in the test split. (default: 2)
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, test_ratio: float = 0.2, min_interactions: int = 2, random_state: int = None, **kwargs):
        super().__init__(**kwargs)
        self.test_ratio = test_ratio
        self.min_interactions = min_interactions
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        print(f"⚙️ Splitting raw data (Test Ratio : {self.test_ratio:.2f}, Min Interactions : {self.min_interactions:,})...")

        # Calculate Interaction Count per User
        user_interaction_counts = df.groupby("user_idx")["product_idx"].count()

        # Separate Eligible & Ineligible Users
        eligible_users = user_interaction_counts[user_interaction_counts >= self.min_interactions].index
        ineligible_users = user_interaction_counts[user_interaction_counts < self.min_interactions].index

        print(f"Eligible Users : {len(eligible_users):,} / {len(user_interaction_counts):,}")
        print(f"Ineligible Users : {len(ineligible_users):,} / {len(user_interaction_counts):,}")

        # Split Users into Eligible & Ineligible
        df_eligible = df[df["user_idx"].isin(eligible_users)]
        df_ineligible = df[df["user_idx"].isin(ineligible_users)]

        # Split Eligible Users into Train & Test
        df_train, df_test = train_test_split(df_eligible, test_size=self.test_ratio, random_state=self.random_state, stratify=df_eligible["user_idx"])

        # Merge Ineligible Users with Train Set
        df_train = pd.concat([df_train, df_ineligible])

        return df_train, df_test
