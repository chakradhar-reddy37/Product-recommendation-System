"""
Product Recommendation System
==============================

A complete collaborative-filtering recommendation project based on
electronics user-rating data.

Implemented:
1. Data loading and preprocessing
2. User-item rating matrix construction
3. User-based collaborative filtering with cosine similarity
4. Top-N personalized recommendations
5. SVD-based matrix factorization
6. Rating prediction
7. RMSE and MAE evaluation on a held-out test set

Expected CSV format:
    user_id,product_id,rating

Example:
    101,501,5
    101,502,4
    102,501,3

Install:
    pip install pandas numpy scikit-learn

Run:
    python "Product recommendation System.py ratings.csv"

If no CSV is supplied, a small demonstration dataset is used.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# DATA LOADING
# ============================================================

def create_demo_data():
    """Create a small electronics-rating dataset for demonstration."""
    rows = [
        (1, "Laptop A", 5), (1, "Phone A", 4), (1, "Headphones A", 5),
        (1, "Monitor A", 3), (2, "Laptop A", 4), (2, "Phone A", 5),
        (2, "Headphones B", 4), (2, "Keyboard A", 5),
        (3, "Laptop B", 5), (3, "Phone B", 4), (3, "Monitor A", 5),
        (3, "Keyboard B", 4), (4, "Phone A", 5), (4, "Headphones A", 4),
        (4, "Keyboard A", 5), (4, "Mouse A", 4),
        (5, "Laptop A", 5), (5, "Phone B", 3), (5, "Monitor A", 4),
        (5, "Mouse A", 5), (6, "Laptop B", 4), (6, "Phone B", 5),
        (6, "Headphones B", 5), (6, "Keyboard B", 4),
        (7, "Phone A", 4), (7, "Headphones A", 5), (7, "Mouse A", 4),
        (7, "Monitor A", 5),
    ]

    return pd.DataFrame(
        rows,
        columns=["user_id", "product_id", "rating"],
    )


def load_data(file_path=None):
    """
    Load ratings from CSV.

    Required columns:
        user_id
        product_id
        rating
    """
    if file_path:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)
    else:
        print("No dataset supplied. Using demonstration data.")
        df = create_demo_data()

    required = {"user_id", "product_id", "rating"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {sorted(required)}"
        )

    df = df[["user_id", "product_id", "rating"]].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["user_id", "product_id", "rating"])

    return df


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_ratings(df, test_size=0.20, random_state=42):
    """Create a held-out test set for objective evaluation."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    return train.reset_index(drop=True), test.reset_index(drop=True)


# ============================================================
# USER-ITEM MATRIX
# ============================================================

def build_user_item_matrix(df):
    """
    Create a user-item matrix.

    Rows    -> users
    Columns -> products
    Values  -> ratings
    """
    return df.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating",
        aggfunc="mean",
    )


# ============================================================
# USER-BASED COLLABORATIVE FILTERING
# ============================================================

class UserBasedCollaborativeFiltering:
    """
    User-based collaborative filtering using cosine similarity.

    For an active user:
        1. Find similar users.
        2. Look at products those users rated.
        3. Aggregate ratings weighted by similarity.
    """

    def __init__(self, train_df):
        self.train_df = train_df
        self.matrix = build_user_item_matrix(train_df)

        # Missing ratings are represented as zero only for similarity
        # computation. The original matrix remains untouched.
        filled = self.matrix.fillna(0)

        self.similarity = cosine_similarity(filled)

        self.similarity_df = pd.DataFrame(
            self.similarity,
            index=self.matrix.index,
            columns=self.matrix.index,
        )

    def predict_rating(self, user_id, product_id, k=10):
        """Predict a user's rating for a product."""
        if user_id not in self.matrix.index:
            return float(self.train_df["rating"].mean())

        if product_id not in self.matrix.columns:
            return float(self.train_df["rating"].mean())

        user_similarities = self.similarity_df.loc[user_id]

        # Users who actually rated the target product.
        ratings = self.matrix[product_id].dropna()

        if ratings.empty:
            return float(self.train_df["rating"].mean())

        similarities = user_similarities.loc[ratings.index]

        # Remove the active user from its own neighborhood.
        if user_id in similarities.index:
            similarities = similarities.drop(user_id, errors="ignore")

        ratings = ratings.loc[similarities.index]

        if len(ratings) == 0:
            return float(self.train_df["rating"].mean())

        # Top-K most similar users.
        top_users = similarities.abs().sort_values(
            ascending=False
        ).head(k)

        similarities = similarities.loc[top_users.index]
        ratings = ratings.loc[top_users.index]

        denominator = similarities.abs().sum()

        if denominator == 0:
            return float(self.train_df["rating"].mean())

        prediction = np.sum(
            similarities.values * ratings.values
        ) / denominator

        return float(np.clip(prediction, 1, 5))

    def recommend(self, user_id, n=5, k=10):
        """
        Generate Top-N recommendations for a user.

        Products already rated by the user are excluded.
        """
        if user_id not in self.matrix.index:
            return pd.DataFrame(
                columns=["product_id", "predicted_rating"]
            )

        rated_products = set(
            self.matrix.loc[user_id]
            .dropna()
            .index
        )

        candidates = [
            product
            for product in self.matrix.columns
            if product not in rated_products
        ]

        predictions = []

        for product in candidates:
            prediction = self.predict_rating(
                user_id,
                product,
                k=k,
            )

            predictions.append(
                (product, prediction)
            )

        recommendations = pd.DataFrame(
            predictions,
            columns=["product_id", "predicted_rating"],
        )

        return recommendations.sort_values(
            "predicted_rating",
            ascending=False,
        ).head(n)


# ============================================================
# SVD MATRIX FACTORIZATION
# ============================================================

class SVDRecommender:
    """
    SVD-based matrix factorization.

    The model learns:
        rating ≈ global_mean + user_bias + item_bias
                 + user_factors · item_factors

    Optimization is performed using stochastic gradient descent.
    """

    def __init__(
        self,
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        epochs=50,
        random_state=42,
    ):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.random_state = random_state

        self.user_to_index = {}
        self.item_to_index = {}

        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0.0

    def fit(self, df):
        rng = np.random.default_rng(self.random_state)

        users = df["user_id"].unique()
        items = df["product_id"].unique()

        self.user_to_index = {
            user: index
            for index, user in enumerate(users)
        }

        self.item_to_index = {
            item: index
            for index, item in enumerate(items)
        }

        n_users = len(users)
        n_items = len(items)

        self.global_mean = float(df["rating"].mean())

        self.user_factors = rng.normal(
            0,
            0.1,
            size=(n_users, self.n_factors),
        )

        self.item_factors = rng.normal(
            0,
            0.1,
            size=(n_items, self.n_factors),
        )

        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        observations = [
            (
                self.user_to_index[row.user_id],
                self.item_to_index[row.product_id],
                float(row.rating),
            )
            for row in df.itertuples()
        ]

        for epoch in range(self.epochs):
            rng.shuffle(observations)

            for user_index, item_index, rating in observations:
                prediction = self._predict_indices(
                    user_index,
                    item_index,
                )

                error = rating - prediction

                # Bias updates.
                self.user_bias[user_index] += (
                    self.learning_rate
                    * (
                        error
                        - self.regularization
                        * self.user_bias[user_index]
                    )
                )

                self.item_bias[item_index] += (
                    self.learning_rate
                    * (
                        error
                        - self.regularization
                        * self.item_bias[item_index]
                    )
                )

                # Save current vectors before simultaneous update.
                user_vector = self.user_factors[user_index].copy()
                item_vector = self.item_factors[item_index].copy()

                self.user_factors[user_index] += (
                    self.learning_rate
                    * (
                        error * item_vector
                        - self.regularization * user_vector
                    )
                )

                self.item_factors[item_index] += (
                    self.learning_rate
                    * (
                        error * user_vector
                        - self.regularization * item_vector
                    )
                )

            if (epoch + 1) % 10 == 0:
                print(
                    f"SVD epoch {epoch + 1}/{self.epochs}"
                )

        return self

    def _predict_indices(self, user_index, item_index):
        prediction = (
            self.global_mean
            + self.user_bias[user_index]
            + self.item_bias[item_index]
            + np.dot(
                self.user_factors[user_index],
                self.item_factors[item_index],
            )
        )

        return float(np.clip(prediction, 1, 5))

    def predict_rating(self, user_id, product_id):
        """Predict a rating for a user-product pair."""
        if user_id not in self.user_to_index:
            return self.global_mean

        if product_id not in self.item_to_index:
            return self.global_mean

        return self._predict_indices(
            self.user_to_index[user_id],
            self.item_to_index[product_id],
        )

    def recommend(self, user_id, all_products, rated_products, n=5):
        """Generate personalized Top-N recommendations."""
        predictions = []

        for product in all_products:
            if product in rated_products:
                continue

            prediction = self.predict_rating(
                user_id,
                product,
            )

            predictions.append(
                (product, prediction)
            )

        result = pd.DataFrame(
            predictions,
            columns=["product_id", "predicted_rating"],
        )

        return result.sort_values(
            "predicted_rating",
            ascending=False,
        ).head(n)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_df, model_name):
    """Evaluate a recommender using RMSE and MAE."""
    actual = []
    predicted = []

    for row in test_df.itertuples():
        prediction = model.predict_rating(
            row.user_id,
            row.product_id,
        )

        actual.append(float(row.rating))
        predicted.append(prediction)

    if not actual:
        return None

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")

    return {
        "model": model_name,
        "RMSE": rmse,
        "MAE": mae,
    }


# ============================================================
# MAIN PROJECT PIPELINE
# ============================================================

def main():
    dataset_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    print("=" * 60)
    print("PRODUCT RECOMMENDATION SYSTEM")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load ratings data
    # --------------------------------------------------------

    df = load_data(dataset_path)

    print("\nDataset preview:")
    print(df.head())

    print(
        f"\nUsers: {df['user_id'].nunique()}"
        f"\nProducts: {df['product_id'].nunique()}"
        f"\nRatings: {len(df)}"
    )

    # --------------------------------------------------------
    # 2. Train/test split
    # --------------------------------------------------------

    train_df, test_df = split_ratings(df)

    print(
        f"\nTraining ratings: {len(train_df)}"
        f"\nTest ratings:     {len(test_df)}"
    )

    # --------------------------------------------------------
    # 3. User-based collaborative filtering
    # --------------------------------------------------------

    collaborative_model = (
        UserBasedCollaborativeFiltering(train_df)
    )

    evaluate_model(
        collaborative_model,
        test_df,
        "User-Based Collaborative Filtering",
    )

    # --------------------------------------------------------
    # 4. SVD matrix factorization
    # --------------------------------------------------------

    svd_model = SVDRecommender(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        epochs=50,
    )

    svd_model.fit(train_df)

    evaluate_model(
        svd_model,
        test_df,
        "SVD Matrix Factorization",
    )

    # --------------------------------------------------------
    # 5. Generate personalized recommendations
    # --------------------------------------------------------

    user_id = train_df["user_id"].iloc[0]

    rated_products = set(
        train_df.loc[
            train_df["user_id"] == user_id,
            "product_id",
        ]
    )

    all_products = df["product_id"].unique()

    print(
        f"\nTop recommendations for user {user_id}"
    )

    print("\nCollaborative Filtering:")
    print(
        collaborative_model.recommend(
            user_id,
            n=5,
            k=10,
        ).to_string(index=False)
    )

    print("\nSVD Matrix Factorization:")
    print(
        svd_model.recommend(
            user_id,
            all_products,
            rated_products,
            n=5,
        ).to_string(index=False)
    )

    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()
