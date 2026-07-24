import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


THEME = {
    "primary": "#00E676",
    "primary_dark": "#00C853",
    "primary_deeper": "#1B5E20",
    "secondary": "#FF6D00",
    "accent": "#00B0FF",
}


def rating_distribution_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="rate", nbins=25,
        color_discrete_sequence=["#00E676"],
        labels={"rate": "Rating", "count": "Count"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20), bargap=0.05,
    )
    return fig


def cost_distribution_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="approx_cost(for two people)", nbins=25,
        color_discrete_sequence=["#FF6D00"],
        labels={"approx_cost(for two people)": "Cost (₹)", "count": "Count"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20), bargap=0.05,
    )
    return fig


def cuisine_distribution_chart(df: pd.DataFrame) -> go.Figure:
    cuisine_counts = (
        df["cuisines"].str.split(",").explode().str.strip()
        .value_counts().head(15)
    )
    fig = px.bar(
        x=cuisine_counts.values, y=cuisine_counts.index, orientation="h",
        color_discrete_sequence=["#00E676"],
        labels={"x": "Count", "y": "Cuisine"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def top_cities_chart(df: pd.DataFrame) -> go.Figure:
    city_counts = df["listed_in(city)"].value_counts().head(15)
    fig = px.bar(
        x=city_counts.index, y=city_counts.values,
        color_discrete_sequence=["#00B0FF"],
        labels={"x": "Area", "y": "Restaurants"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        xaxis=dict(tickangle=-45, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def online_order_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["online_order"].value_counts()
    fig = px.pie(
        values=counts.values, names=["Available", "Not Available"],
        color_discrete_sequence=["#00E676", "#FF1744"], hole=0.5,
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(font=dict(color="#e0e0e0")),
    )
    return fig


def table_booking_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["book_table"].value_counts()
    fig = px.pie(
        values=counts.values, names=["Available", "Not Available"],
        color_discrete_sequence=["#00B0FF", "#FFD600"], hole=0.5,
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(font=dict(color="#e0e0e0")),
    )
    return fig


def top_restaurants_chart(df: pd.DataFrame) -> go.Figure:
    top = (
        df.groupby("name").agg({"rate": "mean", "votes": "sum"})
        .sort_values("rate", ascending=False).head(10).reset_index()
    )
    fig = px.bar(
        top, x="name", y="rate", color="votes",
        color_continuous_scale=[[0, "#1B5E20"], [0.5, "#00E676"], [1, "#FFD600"]],
        labels={"name": "Restaurant", "rate": "Rating", "votes": "Votes"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        xaxis=dict(tickangle=-45, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def rating_vs_cost_chart(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="approx_cost(for two people)", y="rate",
        color="listed_in(type)", size="votes",
        hover_data=["name", "cuisines"],
        labels={"approx_cost(for two people)": "Cost for Two (₹)", "rate": "Rating"},
        opacity=0.7,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(font=dict(color="#e0e0e0", size=10)),
    )
    return fig


def popular_restaurants_chart(df: pd.DataFrame) -> go.Figure:
    popular = df.sort_values("votes", ascending=False).head(10)
    fig = px.bar(
        popular, x="name", y="votes", color="rate",
        color_continuous_scale="Greens",
        labels={"name": "Restaurant", "votes": "Votes", "rate": "Rating"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        xaxis=dict(tickangle=-45, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def listed_type_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["listed_in(type)"].value_counts()
    fig = px.pie(
        values=counts.values, names=counts.index,
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.45,
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(font=dict(color="#e0e0e0")),
    )
    return fig


def rest_type_chart(df: pd.DataFrame) -> go.Figure:
    counts = (
        df["rest_type"].str.split(",").explode().str.strip()
        .value_counts().head(10)
    )
    fig = px.bar(
        x=counts.values, y=counts.index, orientation="h",
        color_discrete_sequence=["#00B0FF"],
        labels={"x": "Count", "y": "Restaurant Type"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def render_restaurant_card(row: pd.Series, idx: int) -> str:
    delay = idx * 0.12

    online_badge = (
        '<span class="badge badge-online">● Online Order</span>'
        if row["online_order"] == 1
        else '<span class="badge badge-offline">○ No Online</span>'
    )
    table_badge = (
        '<span class="badge badge-table">⊞ Table Booking</span>'
        if row["book_table"] == 1
        else ""
    )

    dish_liked = str(row.get("dish_liked", "")).strip()
    dish_section = ""
    if dish_liked and dish_liked != "nan" and dish_liked != "":
        dish_section = f'<p class="rec-card-cuisine">👨‍🍳 {dish_liked[:100]}{"..." if len(dish_liked) > 100 else ""}</p>'

    listed_type = str(row.get("listed_in(type)", ""))
    type_section = ""
    if listed_type and listed_type != "nan":
        type_section = f'<span class="badge badge-type">📂 {listed_type}</span>'

    city_area = str(row.get("listed_in(city)", row.get("location", "")))

    card = f"""
    <div class="rec-card" style="animation-delay:{delay}s">
        <div class="rec-card-glow"></div>
        <div class="rec-card-inner">
            <div class="rec-card-header">
                <div class="rec-badge-wrap">
                    <span class="rec-badge">
                        <span class="rec-badge-icon">✦</span> RECOMMENDED
                    </span>
                </div>
                <div class="rec-rating-box">
                    <span class="rec-rating-star">★</span>
                    <span class="rec-rating-val">{row['rate']:.1f}</span>
                    <span class="rec-rating-count">{int(row['votes'])} votes</span>
                </div>
            </div>
            <h3 class="rec-card-title">{row['name']}</h3>
            <p class="rec-card-location"><span class="rec-loc-icon">◉</span> {row['location']} · {city_area}</p>
            <p class="rec-card-cuisine">🍽 {row['cuisines']}</p>
            <p class="rec-card-type">📋 {row['rest_type']}</p>
            {dish_section}
            <div class="rec-card-badges">
                {online_badge} {table_badge} {type_section}
            </div>
            <div class="rec-card-footer">
                <span class="rec-price">₹{int(row['approx_cost(for two people)'])}</span>
                <span class="rec-price-label">for two</span>
            </div>
        </div>
    </div>
    """
    return card


def metric_card_3d(title: str, value, icon: str, color: str = "#00E676") -> str:
    return f"""
    <div class="metric-3d">
        <div class="metric-3d-shine"></div>
        <div class="metric-3d-icon">{icon}</div>
        <div class="metric-3d-value" style="color:{color};">{value}</div>
        <div class="metric-3d-label">{title}</div>
    </div>
    """
