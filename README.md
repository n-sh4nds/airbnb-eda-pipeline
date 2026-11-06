# Airbnb Exploratory Data Analysis (EDA)

## Project Overview🌐
This project prepares an Airbnb dataset for predictive modeling (predicting price per night). The script handles data cleaning, outlier detection, and feature encoding using Python.

## What the Code Does📜
*   **Data Cleaning:** Renames columns for readability, drops irrelevant features (like IDs and names), and standardizes data types (converting strings to floats & datetimes).
*   **Outlier Detection:** Uses **Z-scores** (via `scipy.stats`) to identify and flag outliers in numerical columns like price, minimum nights, & yearly availability.
*   **Feature Engineering:** Applies **One-Hot Encoding** (`pd.get_dummies`) to categorical variables like neighborhood group & room type.
*   **Visualization:** Generates scatterplots for outliers and a **Pearson Correlation Heatmap** using `seaborn` to identify highly correlated features.

## Libraries Used📚
*   **Pandas & NumPy:** Data manipulation and cleaning
*   **SciPy:** Statistical analysis (Z-scores)
*   **Seaborn & Matplotlib:** Data visualization
