import pandas as pd

df = pd.read_csv("data/raw/twcs.csv", nrows=20)

print("=" * 70)
print("TWCS SCHEMA DIAGNOSTIC")
print("=" * 70)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nFirst 10 rows:")
print(df[
    ["tweet_id", "author_id", "inbound",
     "response_tweet_id", "in_response_to_tweet_id", "text"]
].head(10).to_string(index=False))

print("\nUnique inbound values:")
print(df["inbound"].value_counts(dropna=False))

print("\nExample tweet IDs:")
print(df["tweet_id"].head(10).tolist())

print("\nExample response_tweet_id values:")
print(df["response_tweet_id"].head(10).tolist())

print("\nExample in_response_to_tweet_id values:")
print(df["in_response_to_tweet_id"].head(10).tolist())
