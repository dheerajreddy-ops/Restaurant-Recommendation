# AI-Based Restaurant Recommendation System

An intelligent restaurant recommendation system built with Streamlit that uses Content-Based Filtering with TF-IDF Vectorization and Cosine Similarity to suggest the best restaurants based on user preferences.

## Features

- **Modern UI** - Professional white and green restaurant-themed interface
- **Interactive Dashboard** - Data analysis with Plotly charts
- **Smart Recommendations** - AI-powered content-based filtering
- **Multi-Factor Filtering** - City, cuisine, budget, rating, online order, table booking
- **Beautiful Restaurant Cards** - Responsive design with gradient badges

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/restaurant-recommendation.git
cd restaurant-recommendation

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Project Structure

```
restaurant_recommendation/
├── app.py                  # Main Streamlit application
├── dataset.csv             # Restaurant dataset
├── recommendation.py       # Recommendation model (TF-IDF + Cosine Similarity)
├── preprocessing.py        # Data cleaning and feature engineering
├── utils.py                # UI helpers and chart generators
├── generate_dataset.py     # Dataset generation script
├── requirements.txt        # Python dependencies
├── assets/                 # Static assets (logo, banner)
└── README.md               # Project documentation
```

## How It Works

1. **Data Loading** - Restaurant dataset is loaded and preprocessed
2. **Data Cleaning** - Missing values handled, duplicates removed
3. **Feature Engineering** - Text features created from cuisine, type, and location
4. **TF-IDF Vectorization** - Text features converted to numerical vectors
5. **Cosine Similarity** - Similarity matrix computed between all restaurants
6. **User Input** - User selects city, cuisine, budget, and other preferences
7. **Filtering + Scoring** - Dataset filtered, composite scores calculated
8. **Top 5 Recommendations** - Best matching restaurants displayed

## Pages

### Home
- Project banner and description
- Key metrics (restaurants, cities, cuisines)
- Workflow diagram
- Dataset overview

### Data Analysis
- Rating distribution
- Cost distribution
- Cuisine distribution
- Online order & table booking analysis
- Top cities and popular restaurants
- Interactive Plotly charts

### Recommendation
- Input widgets for preferences
- AI-powered top 5 recommendations
- Beautiful restaurant cards with details

### About
- Problem statement and objectives
- Technologies used
- Machine learning workflow

## Technologies

- **Python** - Core language
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - TF-IDF, Cosine Similarity
- **Streamlit** - Web framework
- **Plotly** - Interactive charts

## Dataset Columns

| Column | Description |
|--------|-------------|
| name | Restaurant name |
| online_order | Online ordering available |
| book_table | Table booking available |
| rate | Restaurant rating (1-5) |
| votes | Number of votes |
| location | Restaurant location |
| rest_type | Restaurant type |
| cuisines | Cuisine types offered |
| approx_cost_for_two | Approximate cost for two people |

## License

MIT License
