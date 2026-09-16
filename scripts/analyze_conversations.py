import pandas as pd
from collections import defaultdict

INPUT = "data/processed/applesupport_tweets.csv"

print("Loading AppleSupport data...")

df = pd.read_csv(
    INPUT,
    dtype={
        "tweet_id": "int64",
        "author_id": "string",
        "inbound": "boolean",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string",
    },
)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    errors="coerce"
)

tweet_ids = set(df["tweet_id"])

# Build parent -> children mapping.
children = defaultdict(list)

for _, row in df.iterrows():
    parent = row["in_response_to_tweet_id"]

    if pd.notna(parent):
        try:
            parent_id = int(float(parent))

            if parent_id in tweet_ids:
                children[parent_id].append(int(row["tweet_id"]))

        except (ValueError, TypeError):
            pass

inbound = df[df["inbound"] == True]

starts = inbound[
    inbound["in_response_to_tweet_id"].isna()
]

print()
print("=" * 70)
print("APPLE SUPPORT CONVERSATION ANALYSIS")
print("=" * 70)

print(f"Total AppleSupport-related tweets: {len(df):,}")
print(f"Customer tweets:                  {len(inbound):,}")
print(f"Customer conversation starters:   {len(starts):,}")

# Map tweet IDs to rows for quick lookup.
id_to_row = df.set_index("tweet_id")

lengths = []

for start_id in starts["tweet_id"]:

    visited = set()
    queue = [int(start_id)]

    while queue:

        current = queue.pop(0)

        if current in visited:
            continue

        visited.add(current)

        for child in children.get(current, []):

            if child not in visited:
                queue.append(child)

    lengths.append(len(visited))

lengths = pd.Series(lengths)

print()
print("Thread length statistics:")
print(f"  Threads:       {len(lengths):,}")
print(f"  Mean:          {lengths.mean():.2f}")
print(f"  Median:        {lengths.median():.0f}")
print(f"  Maximum:       {lengths.max():.0f}")

print()
print("Thread categories:")

single = (lengths == 1).sum()
short = ((lengths >= 2) & (lengths <= 4)).sum()
medium = ((lengths >= 5) & (lengths <= 10)).sum()
long = (lengths >= 11).sum()

print(f"  Single tweet:  {single:,}")
print(f"  2-4 tweets:    {short:,}")
print(f"  5-10 tweets:   {medium:,}")
print(f"  11+ tweets:    {long:,}")

brand_replies = df[
    (df["author_id"] == "AppleSupport") &
    (df["inbound"] == False)
]

parent_ids = pd.to_numeric(
    brand_replies["in_response_to_tweet_id"],
    errors="coerce"
).dropna().astype("int64")

customer_parents = df[
    (df["tweet_id"].isin(parent_ids)) &
    (df["inbound"] == True)
]

print()
print(f"Direct customer -> AppleSupport pairs: {len(customer_parents):,}")

print()
print("=" * 70)
print("SAMPLE CONVERSATIONS")
print("=" * 70)

shown = 0

for start_id in starts["tweet_id"]:

    if shown >= 5:
        break

    queue = [int(start_id)]
    visited = set()
    conversation = []

    while queue:

        current = queue.pop(0)

        if current in visited:
            continue

        visited.add(current)

        if current in id_to_row.index:
            row = id_to_row.loc[current]
            conversation.append(row)

        for child in children.get(current, []):

            if child not in visited:
                queue.append(child)

    if len(conversation) >= 2:

        conversation.sort(
            key=lambda x: x["created_at"]
        )

        print()
        print(f"--- Conversation {shown + 1} ---")

        for message in conversation[:10]:

            speaker = (
                "APPLE"
                if message["inbound"] == False
                else "CUSTOMER"
            )

            text = str(message["text"]).replace("\n", " ")

            print(f"[{speaker}] {text}")

        shown += 1
