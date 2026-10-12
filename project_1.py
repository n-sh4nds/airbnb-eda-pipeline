# Name: Nadia Shands-Speight
# Date: 10/5/2026
# Program Name: Project 1 - Exploratory Data Analysis (EDA)
# Description: This program prepares a CSV file (airbnb.csv) for modeling so that we can predict the price per night of an Airbnb rental based on the factors available in the dataset.

import pandas as pd
import numpy as np
from scipy import stats

def main():

    airbnb_df = pd.read_csv("airbnb_data.csv") # read CSV file into DataFrame

    airbnb_df.rename(columns={ # rename columns for better readability
        'id': 'listing_id', # DROPPED due to relevance!
        'NAME': 'listing_name', # DROPPED due to relevance!
        'host id': 'host_id', # DROPPED due to relevance!
        'host_identity_verified': 'host_verified', # one-hot encoded
        'host name': 'host_name', # DROPPED due to relevance!
        'neighbourhood group': 'nbh_group', # one-hot encoded
        'neighbourhood': 'nbh', # ?????
        'lat': 'latitude', # already clean
        'long': 'longitude', # CLEANED
        'country': 'country', # DROPPED due to relevance!
        'country code': 'country_code', # DROPPED due to relevance!
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
        'house_rules': 'house_rules', # DROPPED due to lack of data and relevance!
        'license': 'license' # DROPPED due to lack of data!
    }, inplace=True)

    print("\nInitial Airbnb Data Summary:") # see summaries of all the features
    print(airbnb_df.describe(include="all").T)
    airbnb_df.info()

    ######### Data Cleaning ############

    airbnb_df = airbnb_df.drop(columns=[ # drop unusable features
        "listing_id", "listing_name", "host_id", "host_name", "country", "country_code", "house_rules", "license"
    ])

    airbnb_df = airbnb_df.drop(index=52885) # exclude 1 heavily misaligned row of data

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

    whole_num_values = ["min_nights", "num_reviews", "host_listings_count", "yearly_availability", "year_built"] # create list of features that have values that logically require whole numbers

    airbnb_df[whole_num_values] = airbnb_df[whole_num_values].apply(
        pd.to_numeric, errors="coerce").astype("Int64") # convert values within whole_num_values to Int64

    monetary_values = ["price", "service_fee"] # create list of features that hold dollar amounts

    airbnb_df[monetary_values] = airbnb_df[monetary_values].apply(
        lambda col: pd.to_numeric( # run this function on columns within monetary_values
            col.str.replace(r"[^0-9.]", "", regex=True), # remove any character from the value that isn't a digit or a period
            errors="coerce"
        )
    )

    ######## Missing and Invalid Values ##########

    invalid_availability = (airbnb_df["yearly_availability"] >= 366) | (airbnb_df["yearly_availability"] < 0) # assign an "invalid" label to all invalid yearly_available values because a year consists of 365 days

    invalid_min_nights = (airbnb_df["min_nights"] < 1) # assign an "invalid" label to all invalid min_night values because an airbnb cannot be booked for less than 1 night

    airbnb_df.loc[invalid_availability, "yearly_availability"] = pd.NA # convert the invalid yearly_availabililty values to NaN
    airbnb_df.loc[invalid_min_nights, "min_nights"] = pd.NA # convert the invalid min_nights values to NaN

    total_invalid_values = int( # add the total sum of all invalid values
        invalid_availability.sum() + invalid_min_nights.sum()
    )

    print("\nInvalid Value Counts (NaN):", total_invalid_values)
    print("yearly_availability:", int(invalid_availability.sum()))
    print("min_nights:", int(invalid_min_nights.sum()))

    print("\nMissing Value Counts:")
    print(airbnb_df.isna().sum().sort_values(ascending=False)) # print a count of missing value for each feature

    ######### Outlier Identification ###########

    airbnb_df["price_z"] = stats.zscore( # calculate z-scores for price column
        airbnb_df["price"],
        nan_policy="omit" # omit any missing values from the calculation
    )

    price_outliers = airbnb_df.query( # create DataFrame to query and find outliers
        "price_z > 3 | price_z < -3" # locate z-scores greater than 3 or less than -3
    )

    airbnb_df["service_fee_z"] = stats.zscore( # calculate z-scores for service_fee column
        airbnb_df["service_fee"],
        nan_policy="omit" # omit any missing values from the calculation
    )

    service_fee_outliers = airbnb_df.query( # create DataFrame to query and find outliers
        "service_fee_z > 3 | service_fee_z < -3" # locate z-scores greater than 3 or less than -3
    )

    airbnb_df["review_rate_z"] = stats.zscore( # calculate z-scores for review_rate column
        airbnb_df["review_rate"],
        nan_policy="omit" # omit any missing values from the calculation
    )

    review_rate_outliers = airbnb_df.query( # create DataFrame to query and find outliers
        "review_rate_z > 3 | review_rate_z < -3" # locate z-scores greater than 3 or less than -3
    )

    airbnb_df["host_listings_count_z"] = stats.zscore( # calculate z-scores for host_listings_count column
        airbnb_df["host_listings_count"],
        nan_policy="omit" # omit any missing values from the calculation
    )

    host_listings_outliers = airbnb_df.query ( # create DataFrame to query and find outliers
        "host_listings_count_z > 3 | host_listings_count_z < -3" # locate z-scores greater than 3 or less than -3
    )

    print("\nPrice Z-Scores:")
    print(airbnb_df[["price", "price_z"]])

    print("\nPrice Outliers:", len(price_outliers))
    print(price_outliers[["price", "price_z"]])

    print("\nService Fee Z-Scores:")
    print(airbnb_df[["service_fee", "service_fee_z"]])

    print("\nService Fee Outliers:", len(service_fee_outliers))
    print(service_fee_outliers[["service_fee", "service_fee_z"]])

    print("\nReview Rate Number Z-Scores:")
    print(airbnb_df[["review_rate", "review_rate_z"]])

    print("\nReview Rate Number Outliers:", len(review_rate_outliers))
    print(review_rate_outliers[["review_rate", "review_rate_z"]])

    print("\nHost Listings Count Z-Scores:")
    print(airbnb_df[["host_listings_count", "host_listings_count_z"]])

    print("\nHost Listings Count Outliers:", len(host_listings_outliers))
    print(host_listings_outliers[["host_listings_count", "host_listings_count_z"]])

    encoded = pd.DataFrame({ # create a Pandas table and use one-hot encoding to represent the categories within these columns numerically
        'host_verified': airbnb_df['host_verified'], # (unconfirmed, verified, nan)
        'instant_book': airbnb_df['instant_book'], # (false, true, <NA>)
        'cancel_policy': airbnb_df['cancel_policy'], # (flexible, moderate, strict, nan)
        'nbh_group': airbnb_df['nbh_group'], # (Bronx, Brooklyn, Manhattan, Queens, Staten Island, nan)
        'room_type': airbnb_df['room_type'] # (Entire home/apt, Hotel room, Private room, Shared room, nan)
    })

    encoded = pd.get_dummies( # convert each category within the columns of encoded into indicator columns
        encoded,
        dummy_na=True # add NaN indicator columns for all the missing values
    )

    airbnb_df = airbnb_df.join(encoded) # join the encoded columns to airbnb_df while retaining the original columns

    print("\nClean Airbnb Data Summary:")
    print(airbnb_df.describe(include="all").T)
    airbnb_df.info() # print a compact summary of the data frame with non-null counts and dtypes

    # print(airbnb_df["reviews_per_month"].head(40))

main()
