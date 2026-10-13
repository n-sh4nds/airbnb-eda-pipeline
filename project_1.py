# Name: Nadia Shands-Speight
# Date: 10/6/2026
# Program Name: Project 1 - Exploratory Data Analysis (EDA)
# Description: This program prepares a CSV file (airbnb.csv) for modeling so that we can predict the price per night of an Airbnb rental based on the factors available in the dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():

    airbnb_df = pd.read_csv("airbnb_data.csv") # read CSV file into DataFrame

    ########## Data Cleaning ##########

    airbnb_df.rename(columns={ # rename columns for better readability
        'id': 'listing_id', # DROPPED due to relevance
        'NAME': 'listing_name', # DROPPED due to relevance
        'host id': 'host_id', # DROPPED due to relevance
        'host_identity_verified': 'host_verified', # one-hot encoded
        'host name': 'host_name', # DROPPED due to relevance
        'neighbourhood group': 'nbh_group', # one-hot encoded
        'neighbourhood': 'nbh', # DROPPED due to high cardinality
        'lat': 'latitude', # CLEAN
        'long': 'longitude', # CLEANED
        'country': 'country', # DROPPED due to relevance
        'country code': 'country_code', # DROPPED due to relevance
        'instant_bookable': 'instant_book', # one-hot encoded
        'cancellation_policy': 'cancel_policy', # one-hot encoded
        'room type': 'room_type', # one-hot encoded
        'Construction year': 'year_built', # CLEANED
        'price': 'price', # CLEANED
        'service fee': 'service_fee', # CLEANED
        'minimum nights': 'min_nights', # CLEANED
        'number of reviews': 'num_reviews', # CLEANED
        'last review': 'last_review_date', # CLEANED
        'reviews per month': 'reviews_per_month', # CLEANED
        'review rate number': 'review_rate', # CLEANED
        'calculated host listings count': 'host_listings_count', # CLEAN
        'availability 365': 'yearly_availability', # CLEAN
        'house_rules': 'house_rules', # DROPPED due to lack of data and relevance
        'license': 'license' # DROPPED due to lack of data
    }, inplace=True)

    airbnb_df = airbnb_df.drop(columns=[ # drop unusable features
        "listing_id", "listing_name", "host_id", "host_name", "nbh", "country", "country_code", "house_rules", "license",
    ])

    airbnb_df = airbnb_df.drop(index=52885) # exclude a heavily misaligned row of data

    airbnb_df["longitude"] = ( # convert longitude values from str to float64
        pd.to_numeric(airbnb_df["longitude"]).astype("float64")
    )

    airbnb_df["last_review_date"] = ( # convert last_review_date values from str to datetime64[us]
        pd.to_datetime(airbnb_df["last_review_date"])
    )

    airbnb_df["instant_book"] = ( # standardize capitalization of instant_book values
        airbnb_df["instant_book"].astype("string").str.strip().str.lower() # convert from object to str and enforce lowercase-only values
    )

    airbnb_df["nbh_group"] = ( # replace misspelled nbh_group values "brookln" and "manhatan" with correct spellings
        airbnb_df["nbh_group"]
        .replace({"brookln": "Brooklyn", "manhatan": "Manhattan"})
    )

    whole_num_values = ["min_nights", "num_reviews", "host_listings_count", "yearly_availability", "year_built"] # create list of features with values that logically require whole numbers

    airbnb_df[whole_num_values] = airbnb_df[whole_num_values].apply(
        pd.to_numeric, errors="coerce").astype("Int64") # convert values within whole_num_values to Int64

    monetary_values = ["price", "service_fee"] # create list of features that hold dollar amounts

    airbnb_df[monetary_values] = airbnb_df[monetary_values].apply(
        lambda col: pd.to_numeric( # run this function on columns within monetary_values
            col.str.replace(r"[^0-9.]", "", regex=True), # remove any character from the value that isn't a numeric digit or a period
            errors="coerce"
        )
    )

    ########## Encoded Features ##########

    encoded = pd.DataFrame({ # create Pandas table and use one-hot encoding to represent categories within these columns numerically
        'host_verified': airbnb_df['host_verified'], # (unconfirmed, verified, nan)
        'instant_book': airbnb_df['instant_book'], # (false, true, <NA>)
        'cancel_policy': airbnb_df['cancel_policy'], # (flexible, moderate, strict, nan)
        'nbh_group': airbnb_df['nbh_group'], # (Bronx, Brooklyn, Manhattan, Queens, Staten Island, nan)
        'room_type': airbnb_df['room_type'] # (Entire home/apt, Hotel room, Private room, Shared room, nan)
    })

    encoded = pd.get_dummies( # convert each category into indicator columns
        encoded,
        dummy_na=True, # add NaN indicator columns for any missing values
        drop_first=True # leave out the first category as the reference level
    )

    airbnb_df = airbnb_df.join(encoded) # join columns from 'encoded' to 'airbnb_df' while retaining original columns

    print("\nClean Airbnb Data Summary:")
    print(airbnb_df.describe(include="all").T)
    airbnb_df.info() # print compact summary of airbnb_df with non-null counts and dtypes

    ########## Invalid Values ##########

    invalid_availability = (airbnb_df["yearly_availability"] >= 366) | (airbnb_df["yearly_availability"] < 0) # label all invalid yearly_available values
    invalid_min_nights = (airbnb_df["min_nights"] < 1) | (airbnb_df["min_nights"] > 365) # label all invalid min_night values

    airbnb_df.loc[invalid_availability, "yearly_availability"] = pd.NA # convert invalid yearly_availabililty values to NaN
    airbnb_df.loc[invalid_min_nights, "min_nights"] = pd.NA # convert invalid min_nights values to NaN

    total_invalid_values = int( # find sum of all invalid values
        invalid_availability.sum() + invalid_min_nights.sum()
    )

    print("\nTotal Invalid Value Count:", total_invalid_values)
    print("--- yearly_availability:", int(invalid_availability.sum()))
    print("--- min_nights:", int(invalid_min_nights.sum()))

    print("\nMissing Value Count:")
    print(airbnb_df.isna().sum().sort_values(ascending=False)) # print a count of missing values for each feature

    ########## Outliers ##########

    airbnb_df["price_z"] = stats.zscore( # calculate z-scores for price column
        airbnb_df["price"],
        nan_policy="omit" # omit any missing values from the calculation
    )

    price_outliers = airbnb_df.query( # create DataFrame to query and find outliers
        "price_z > 3 | price_z < -3" # locate z-scores greater than 3 or less than -3
    )

    airbnb_df["service_fee_z"] = stats.zscore(
        airbnb_df["service_fee"],
        nan_policy="omit"
    )

    service_fee_outliers = airbnb_df.query(
        "service_fee_z > 3 | service_fee_z < -3"
    )

    airbnb_df["host_listings_count_z"] = stats.zscore(
        airbnb_df["host_listings_count"],
        nan_policy="omit"
    )

    host_listings_outliers = airbnb_df.query (
        "host_listings_count_z > 3 | host_listings_count_z < -3"
    )

    airbnb_df["min_nights_z"] = stats.zscore(
        airbnb_df["min_nights"],
        nan_policy="omit"
    )

    min_nights_outliers = airbnb_df.query (
        "min_nights_z > 3 | min_nights_z < -3"
    )

    airbnb_df["yearly_availability_z"] = stats.zscore(
        airbnb_df["yearly_availability"],
        nan_policy="omit"
    )

    yearly_availability_outliers = airbnb_df.query (
        "yearly_availability_z > 3 | yearly_availability_z < -3"
    )

    airbnb_df["year_built_z"] = stats.zscore(
        airbnb_df["year_built"],
        nan_policy="omit"
    )

    year_built_outliers = airbnb_df.query (
        "year_built_z > 3 | year_built_z < -3"
    )

    airbnb_df["num_reviews_z"] = stats.zscore(
        airbnb_df["num_reviews"],
        nan_policy="omit"
    )

    num_reviews_outliers = airbnb_df.query(
        "num_reviews_z > 3 | num_reviews_z < -3"
    )

    airbnb_df["reviews_per_month_z"] = stats.zscore( #
        airbnb_df["reviews_per_month"],
        nan_policy="omit"
    )

    reviews_per_month_outliers = airbnb_df.query (
        "reviews_per_month_z > 3 | reviews_per_month_z < -3"
    )

    airbnb_df["review_rate_z"] = stats.zscore(
        airbnb_df["review_rate"],
        nan_policy="omit"
    )

    review_rate_outliers = airbnb_df.query(
        "review_rate_z > 3 | review_rate_z < -3" #
    )

    print("\nIndex | Price | Z-Score [", len(price_outliers), "total outliers ]")
    print(airbnb_df[["price", "price_z"]])

    print("\nIndex | Service Fee | Z-Score [", len(service_fee_outliers), "total outliers ]")
    print(airbnb_df[["service_fee", "service_fee_z"]])

    print("\nIndex | Host Listings Count | Z-Score [", len(host_listings_outliers), "total outliers ]")
    print(airbnb_df[["host_listings_count", "host_listings_count_z"]])

    print("\nIndex | Minimum Nights | Z-Score [", len(min_nights_outliers), "total outliers ]")
    print(airbnb_df[["min_nights", "min_nights_z"]])

    print("\nIndex | Yearly Availability | Z-Score [", len(yearly_availability_outliers), "total outliers ]")
    print(airbnb_df[["yearly_availability", "yearly_availability_z"]])

    print("\nIndex | Year Built | Z-Score [", len(year_built_outliers), "total outliers ]")
    print(airbnb_df[["year_built", "year_built_z"]])

    print("\nIndex | Number of Reviews | Z-Score [", len(num_reviews_outliers), "total outliers ]")
    print(airbnb_df[["num_reviews", "num_reviews_z"]])

    print("\nIndex | Reviews Per Month | Z-Score [", len(reviews_per_month_outliers), "total outliers ]")
    print(airbnb_df[["reviews_per_month", "reviews_per_month_z"]])

    print("\nIndex | Review Rate Number | Z-Score [", len(review_rate_outliers), "total outliers ]")
    print(airbnb_df[["review_rate", "review_rate_z"]])

    print("\nOutlier Scatterplots [png]")
    plt.scatter(airbnb_df.index, airbnb_df["min_nights_z"], s=50, alpha=0.45) # create scatterplot for min_night_z outliers
    plt.xlabel("Listing Index #")
    plt.ylabel("Minimum Nights Z-Scores")
    plt.axhline(y=3, color="red", linestyle="--")
    plt.axhline(y=-3, color="red", linestyle="--")
    plt.show()

    plt.scatter(airbnb_df.index, airbnb_df["host_listings_count_z"], s=50, alpha=0.45) # create scatterplot for host_listings_count_z outliers
    plt.xlabel("Listing Index #")
    plt.ylabel("Host Listings Count Z-Scores")
    plt.axhline(y=3, color="red", linestyle="--")
    plt.axhline(y=-3, color="red", linestyle="--")
    plt.show()

    plt.scatter(airbnb_df.index, airbnb_df["num_reviews_z"], s=50, alpha=0.45) # create scatterplot for num_reviews_z outliers
    plt.xlabel("Listing Index #")
    plt.ylabel("Number of Reviews Z-Scores")
    plt.axhline(y=3, color="red", linestyle="--")
    plt.axhline(y=-3, color="red", linestyle="--")
    plt.show()

    plt.scatter(airbnb_df.index, airbnb_df["reviews_per_month_z"], s=50, alpha=0.45) # create scatterplot for reviews_per_month_z outliers
    plt.xlabel("Listing Index #")
    plt.ylabel("Reviews per Month Z-Scores")
    plt.axhline(y=3, color="red", linestyle="--")
    plt.axhline(y=-3, color="red", linestyle="--")
    plt.show()

    ########## Pearson Correlation Coefficients ##########

    numeric_features = [ # create list of all usable numeric features (besides price)
        "latitude",
        "longitude",
        "service_fee",
        "min_nights",
        "num_reviews",
        "reviews_per_month",
        "review_rate",
        "host_listings_count",
        "yearly_availability",
        "year_built",
    ]

    correlation_data = pd.concat([ # create new DataFrame that combines numeric values with encoded categorical columns
        airbnb_df[numeric_features].apply(pd.to_numeric, errors="coerce"),
        encoded.astype(int)
    ],axis=1)

    correlation_matrix = correlation_data.corr() # find Pearson correlation coefficients of 'numeric_features' and 'encoded'

    print("\nPearson Correlation Coefficients:")
    print(correlation_matrix.round(2)) # round each coefficient in the matrix to 2 decimal places

    print("\nHighly Correlated Features Heatmap [png]")
    sns.heatmap(correlation_matrix, linewidth=.5, cmap='RdPu') # create the heatmap for the coefficients
    plt.tight_layout()
    plt.show()

main()
