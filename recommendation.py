"""
recommendation.py
Loads the deployed TF-IDF model + sparse matrix from disk
and computes similarity on-the-fly to save memory.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.joblib")
DATAFRAME_PATH = os.path.join(MODEL_DIR, "processed_dataframe.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


class RestaurantRecommender:
    def __init__(self, df=None):
        if df is not None:
            self._train_from_dataframe(df)
        else:
            self._load_deployed_model()

    def _train_from_dataframe(self, df):
        self.df = df.copy()
        self.tfidf = __import__("sklearn.feature_extraction.text", fromlist=["TfidfVectorizer"]).TfidfVectorizer(
            max_features=5000, stop_words="english",
            ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["combined_features"].fillna(""))
        self.metadata = None
        self.model_loaded = "trained"

    def _load_deployed_model(self):
        if not os.path.exists(TFIDF_MATRIX_PATH):
            raise FileNotFoundError(
                f"No deployed model found at {MODEL_DIR}. "
                "Run 'python train_model.py' first."
            )

        self.tfidf = joblib.load(VECTORIZER_PATH)
        self.tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
        self.df = joblib.load(DATAFRAME_PATH)

        with open(METADATA_PATH) as f:
            self.metadata = json.load(f)

        self.model_loaded = "deployed"

    def _get_avg_similarity(self, indices):
        if not indices:
            return []
        sub_matrix = self.tfidf_matrix[indices]
        sim = cosine_similarity(sub_matrix, self.tfidf_matrix)
        avg_sims = []
        for i in range(len(indices)):
            row = sim[i]
            top5_idx = np.argsort(row)[::-1][1:6]
            avg_sims.append(float(np.mean(row[top5_idx])))
        return avg_sims

    def get_recommendations(
        self,
        city: str = "All",
        cuisine: str = "All",
        budget: int = 3000,
        min_rating: float = 3.0,
        online_order: bool = False,
        table_booking: bool = False,
        top_n: int = 5,
    ) -> pd.DataFrame:
        filtered = self.df.copy()

        if city and city != "All":
            filtered = filtered[
                filtered["listed_in(city)"].str.contains(city, case=False, na=False)
            ]

        if cuisine and cuisine != "All":
            filtered = filtered[
                filtered["cuisines"].str.contains(cuisine, case=False, na=False)
            ]

        if budget is not None:
            filtered = filtered[
                filtered["approx_cost(for two people)"] <= budget
            ]

        if min_rating is not None:
            filtered = filtered[filtered["rate"] >= min_rating]

        if online_order:
            filtered = filtered[filtered["online_order"] == 1]

        if table_booking:
            filtered = filtered[filtered["book_table"] == 1]

        if filtered.empty:
            return pd.DataFrame()

        filtered_indices = filtered.index.tolist()
        avg_similarities = self._get_avg_similarity(filtered_indices)

        filtered = filtered.copy()
        filtered["similarity_score"] = avg_similarities

        max_cost = filtered["approx_cost(for two people)"].max()
        max_votes = filtered["votes"].max()
        if max_cost == 0:
            max_cost = 1
        if max_votes == 0:
            max_votes = 1

        filtered["composite_score"] = (
            0.35 * (filtered["rate"] / 5.0)
            + 0.25 * filtered["similarity_score"]
            + 0.20 * (1 - filtered["approx_cost(for two people)"] / max_cost)
            + 0.20 * (filtered["votes"] / max_votes)
        )

        results = (
            filtered.sort_values("composite_score", ascending=False)
            .head(top_n)[
                [
                    "name", "cuisines", "rate", "votes",
                    "approx_cost(for two people)", "online_order",
                    "book_table", "rest_type", "location",
                    "listed_in(city)", "listed_in(type)", "dish_liked",
                ]
            ]
            .reset_index(drop=True)
        )

        return results

    def get_similar_restaurants(
        self, restaurant_name: str, top_n: int = 5
    ) -> pd.DataFrame:
        idx_matches = self.df[
            self.df["name"].str.contains(restaurant_name, case=False, na=False)
        ].index

        if len(idx_matches) == 0:
            return pd.DataFrame()

        idx = idx_matches[0]
        query = self.tfidf_matrix[idx : idx + 1]
        sim = cosine_similarity(query, self.tfidf_matrix).flatten()
        top_indices = np.argsort(sim)[::-1][1 : top_n + 1]

        return self.df.iloc[top_indices][
            ["name", "cuisines", "rate", "votes", "approx_cost(for two people)", "location"]
        ]

    def get_model_info(self) -> dict:
        if self.model_loaded == "deployed" and self.metadata:
            return {
                "source": "deployed",
                "path": MODEL_DIR,
                **self.metadata,
            }
        return {"source": "trained_in_memory", "restaurants": len(self.df)}
