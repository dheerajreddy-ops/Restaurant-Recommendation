"""
train_model.py
Trains the TF-IDF model and saves sparse matrix + dataframe to disk.
No dense similarity matrix — computed on-the-fly to save memory.

Run:  python train_model.py
"""

import os
import sys
import io
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.joblib")
DATAFRAME_PATH = os.path.join(MODEL_DIR, "processed_dataframe.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


def load_raw_data():
    print("[1/6] Loading raw dataset...")
    df = pd.read_csv(DATASET_PATH, encoding="latin-1")
    print(f"      Loaded {len(df):,} rows x {len(df.columns)} columns")
    return df


def clean_data(df):
    print("[2/6] Cleaning data...")
    df = df.copy()

    df.drop_duplicates(subset=["name", "location", "cuisines"], keep="first", inplace=True)
    print(f"      Removed duplicates -> {len(df):,} unique restaurants")

    df["rate"] = df["rate"].astype(str).str.replace(r"/5", "", regex=True).str.strip()
    df["rate"] = df["rate"].replace(["NEW", "-", "nan", "None"], np.nan)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")

    df["approx_cost(for two people)"] = (
        df["approx_cost(for two people)"]
        .astype(str).str.replace(",", "", regex=False).str.strip()
    )
    df["approx_cost(for two people)"] = pd.to_numeric(
        df["approx_cost(for two people)"], errors="coerce"
    )

    df.dropna(subset=["name", "location"], inplace=True)

    df["rate"] = df["rate"].fillna(df["rate"].median())
    df["votes"] = df["votes"].fillna(0).astype(int)
    df["approx_cost(for two people)"] = df["approx_cost(for two people)"].fillna(
        df["approx_cost(for two people)"].median()
    )
    df["rate"] = df["rate"].clip(1.0, 5.0)

    df["online_order"] = (
        df["online_order"].astype(str).str.strip()
        .map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    )
    df["book_table"] = (
        df["book_table"].astype(str).str.strip()
        .map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    )

    df["cuisines"] = df["cuisines"].fillna("Not Specified").str.strip()
    df["rest_type"] = df["rest_type"].fillna("Unknown").str.strip()
    df["location"] = df["location"].str.strip()
    df["listed_in(city)"] = df["listed_in(city)"].fillna(df["location"]).str.strip()
    df["listed_in(type)"] = df["listed_in(type)"].fillna("Other").str.strip()
    df["dish_liked"] = df["dish_liked"].fillna("").str.strip()

    df.reset_index(drop=True, inplace=True)
    print(f"      Cleaned: {len(df):,} restaurants ready")
    return df


def engineer_features(df):
    print("[3/6] Engineering features...")
    df = df.copy()

    df["cuisines_cleaned"] = df["cuisines"].str.replace(",", " ", regex=False).str.lower()
    df["rest_type_cleaned"] = df["rest_type"].str.lower()
    df["location_cleaned"] = df["location"].str.lower()
    df["dish_liked_cleaned"] = df["dish_liked"].str.replace(",", " ", regex=False).str.lower()
    df["listed_type_cleaned"] = df["listed_in(type)"].str.lower()

    df["combined_features"] = (
        df["cuisines_cleaned"] + " "
        + df["rest_type_cleaned"] + " "
        + df["location_cleaned"] + " "
        + df["dish_liked_cleaned"] + " "
        + df["listed_type_cleaned"] + " "
        + df["rate"].astype(str) + " "
        + df["approx_cost(for two people)"].astype(str)
    )

    feature_count = df["combined_features"].apply(lambda x: len(x.split())).mean()
    print(f"      Created combined_features (avg {feature_count:.0f} tokens/sample)")
    return df


def train_tfidf(df):
    print("[4/6] Training TF-IDF Vectorizer...")
    start = time.time()

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    tfidf_matrix = tfidf.fit_transform(df["combined_features"].fillna(""))
    elapsed = time.time() - start

    print(f"      TF-IDF matrix: {tfidf_matrix.shape[0]:,} restaurants x {tfidf_matrix.shape[1]:,} features")
    print(f"      Sparse density: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]) * 100:.1f}%")
    print(f"      Training time: {elapsed:.2f}s")
    return tfidf, tfidf_matrix


def save_model(df, tfidf, tfidf_matrix):
    print("[5/6] Saving model artifacts to disk...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(tfidf, VECTORIZER_PATH, compress=3)
    print(f"      -> Vectorizer: {VECTORIZER_PATH}")

    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH, compress=3)
    print(f"      -> TF-IDF matrix (sparse): {TFIDF_MATRIX_PATH}")

    joblib.dump(df, DATAFRAME_PATH, compress=3)
    print(f"      -> DataFrame: {DATAFRAME_PATH}")

    cities = sorted(df["listed_in(city)"].unique().tolist())
    all_cuisines = set()
    for c in df["cuisines"]:
        for item in str(c).split(","):
            item = item.strip()
            if item and item != "Not Specified":
                all_cuisines.add(item)
    cuisines = sorted(all_cuisines)

    metadata = {
        "dataset": "zomato_bangalore",
        "total_restaurants": len(df),
        "total_cities": len(cities),
        "total_cuisines": len(cuisines),
        "avg_rating": round(float(df["rate"].mean()), 2),
        "avg_cost": round(float(df["approx_cost(for two people)"].mean()), 0),
        "online_order_pct": round(float(df["online_order"].mean() * 100), 1),
        "table_booking_pct": round(float(df["book_table"].mean() * 100), 1),
        "tfidf_features": int(tfidf_matrix.shape[1]),
        "cities": cities,
        "cuisines": cuisines,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"      -> Metadata: {METADATA_PATH}")

    total_size = sum(
        os.path.getsize(p)
        for p in [VECTORIZER_PATH, TFIDF_MATRIX_PATH, DATAFRAME_PATH, METADATA_PATH]
    )
    print(f"      Total model size: {total_size / (1024*1024):.1f} MB")


def verify_model():
    print("[6/6] Verifying deployed model...")
    tfidf = joblib.load(VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    df = joblib.load(DATAFRAME_PATH)

    with open(METADATA_PATH) as f:
        meta = json.load(f)

    assert tfidf_matrix.shape[0] == len(df), "Shape mismatch!"

    test_cuisine = "Italian"
    mask = df["cuisines"].str.contains(test_cuisine, case=False, na=False)
    assert mask.sum() > 0, f"No restaurants found for {test_cuisine}"

    print(f"      [OK] Vectorizer loads OK ({tfidf_matrix.shape[1]} features)")
    print(f"      [OK] TF-IDF matrix loads OK (sparse {tfidf_matrix.shape})")
    print(f"      [OK] DataFrame loads OK ({len(df):,} restaurants)")
    print(f"      [OK] Test query for '{test_cuisine}': {mask.sum()} matches")
    print()
    print("=" * 60)
    print("  MODEL DEPLOYED SUCCESSFULLY!")
    print(f"  Restaurants: {meta['total_restaurants']:,}")
    print(f"  Areas: {meta['total_cities']}")
    print(f"  Cuisines: {meta['total_cuisines']}")
    print(f"  TF-IDF Features: {meta['tfidf_features']}")
    print(f"  Model Dir: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("  AI RESTAURANT RECOMMENDER — MODEL TRAINING")
    print("=" * 60)
    print()

    total_start = time.time()

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    tfidf, tfidf_matrix = train_tfidf(df)
    save_model(df, tfidf, tfidf_matrix)
    verify_model()

    total_elapsed = time.time() - total_start
    print(f"\nTotal pipeline time: {total_elapsed:.2f}s")
    print("Ready to serve predictions!")
