import pandas as pd
import numpy as np


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin-1")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.drop_duplicates(subset=["name", "location", "cuisines"], keep="first", inplace=True)

    df["rate"] = (
        df["rate"]
        .astype(str)
        .str.replace(r"/5", "", regex=True)
        .str.strip()
    )
    df["rate"] = df["rate"].replace(["NEW", "-", "nan", "None"], np.nan)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")

    df["approx_cost(for two people)"] = (
        df["approx_cost(for two people)"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
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
        df["online_order"].astype(str).str.strip().map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    )
    df["book_table"] = (
        df["book_table"].astype(str).str.strip().map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    )

    df["cuisines"] = df["cuisines"].fillna("Not Specified").str.strip()
    df["rest_type"] = df["rest_type"].fillna("Unknown").str.strip()
    df["location"] = df["location"].str.strip()
    df["listed_in(city)"] = df["listed_in(city)"].fillna(df["location"]).str.strip()
    df["listed_in(type)"] = df["listed_in(type)"].fillna("Other").str.strip()
    df["dish_liked"] = df["dish_liked"].fillna("").str.strip()

    df.reset_index(drop=True, inplace=True)
    return df


def create_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["cuisines_cleaned"] = (
        df["cuisines"].str.replace(",", " ", regex=False).str.lower()
    )
    df["rest_type_cleaned"] = df["rest_type"].str.lower()
    df["location_cleaned"] = df["location"].str.lower()
    df["dish_liked_cleaned"] = (
        df["dish_liked"].str.replace(",", " ", regex=False).str.lower()
    )
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

    return df


def get_dataset_stats(df: pd.DataFrame) -> dict:
    cities = sorted(df["listed_in(city)"].unique().tolist())
    all_cuisines = set()
    for c in df["cuisines"]:
        for item in str(c).split(","):
            item = item.strip()
            if item and item != "Not Specified":
                all_cuisines.add(item)
    cuisines = sorted(all_cuisines)

    rest_types = sorted(df["rest_type"].unique().tolist())
    listed_types = sorted(df["listed_in(type)"].unique().tolist())

    return {
        "total_restaurants": len(df),
        "total_cities": len(cities),
        "total_cuisines": len(cuisines),
        "avg_rating": round(df["rate"].mean(), 2),
        "avg_cost": round(df["approx_cost(for two people)"].mean(), 0),
        "online_order_pct": round(df["online_order"].mean() * 100, 1),
        "table_booking_pct": round(df["book_table"].mean() * 100, 1),
        "cities": cities,
        "cuisines": cuisines,
        "rest_types": rest_types,
        "listed_types": listed_types,
    }


def preprocess_pipeline(filepath: str) -> tuple:
    df = load_dataset(filepath)
    df = clean_dataset(df)
    df = create_text_features(df)
    return df, get_dataset_stats(df)
