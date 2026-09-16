import pandas as pd
from collections import deque
from pathlib import Path

INPUT = Path("data/raw/twcs.csv")
OUTPUT = Path("data/processed/applesupport_conversations.csv")

BRAND = "AppleSupport"

print("Loading original TWCS dataset...")
df = pd.read_csv(INPUT)

print(f"Total dataset rows: {len(df):,}")

# Normalize tweet IDs consistently.
df["tweet_id"] = df["tweet_id"].astype(str)

# Convert parent IDs such as 123.0 -> 123
df["parent_id"] = (
    pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .fillna("")
)

# Identify all AppleSupport tweets
apple_ids = set(
    df.loc[df["author_id"] == BRAND, "tweet_id"]
)

print(f"AppleSupport tweets: {len(apple_ids):,}")

# Find tweets whose parent is an AppleSupport tweet
direct_reply_mask = df["parent_id"].isin(apple_ids)
direct_reply_ids = set(
    df.loc[direct_reply_mask, "tweet_id"]
)

print(f"Direct replies to AppleSupport: {len(direct_reply_ids):,}")

# We want the complete conversation chain.
# First include AppleSupport tweets + direct replies.
relevant_ids = apple_ids | direct_reply_ids

# Build parent lookup for the entire dataset.
parent_lookup = dict(
    zip(df["tweet_id"], df["parent_id"])
)

# Walk backwards through parent links.
queue = deque(direct_reply_ids)
visited = set(relevant_ids)

while queue:
    tweet_id = queue.popleft()
    parent_id = parent_lookup.get(tweet_id, "")

    if parent_id and parent_id not in visited:
        visited.add(parent_id)
        queue.append(parent_id)

print(f"Backward conversation closure: {len(visited):,}")

# Extract all recovered tweets
result = df[df["tweet_id"].isin(visited)].copy()

# Parse dates
result["created_at"] = pd.to_datetime(
    result["created_at"],
    utc=True,
    format="mixed"
)

# ------------------------------------------------------------------
# Build conversation IDs
# ------------------------------------------------------------------

id_set = set(result["tweet_id"])

parent_lookup_result = dict(
    zip(result["tweet_id"], result["parent_id"])
)

root_cache = {}

def find_root(tweet_id):
    if tweet_id in root_cache:
        return root_cache[tweet_id]

    path = []
    current = tweet_id
    seen = set()

    while current and current not in seen:

        if current in root_cache:
            root = root_cache[current]
            break

        seen.add(current)
        path.append(current)

        parent = parent_lookup_result.get(current, "")

        if not parent or parent not in id_set:
            root = current
            break

        current = parent

    else:
        root = current

    for item in path:
        root_cache[item] = root

    return root

result["conversation_id"] = result["tweet_id"].map(find_root)

# Sort conversations chronologically
result = result.sort_values(
    ["conversation_id", "created_at"]
)

# Save
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

sizes = result.groupby("conversation_id").size()

print()
print("=" * 70)
print("CORRECTED APPLESUPPORT CONVERSATION ANALYSIS")
print("=" * 70)

print(f"Saved file: {OUTPUT}")
print(f"Recovered tweets: {len(result):,}")
print(f"Conversations: {result['conversation_id'].nunique():,}")

print()
print("Conversation length statistics:")
print(f"  Mean:    {sizes.mean():.2f}")
print(f"  Median:  {sizes.median():.0f}")
print(f"  Maximum: {sizes.max():.0f}")

print()
print("Conversation categories:")
print(f"  1 tweet:     {(sizes == 1).sum():,}")
print(f"  2-4 tweets:  {sizes.between(2, 4).sum():,}")
print(f"  5-10 tweets: {sizes.between(5, 10).sum():,}")
print(f"  11+ tweets:  {(sizes >= 11).sum():,}")

print()
print("Tweet direction:")
print(f"  Customer tweets: {(result['inbound'] == True).sum():,}")
print(f"  Brand tweets:    {(result['inbound'] == False).sum():,}")

# ------------------------------------------------------------------
# Show a few real multi-turn conversations
# ------------------------------------------------------------------

print()
print("=" * 70)
print("SAMPLE MULTI-TURN CONVERSATIONS")
print("=" * 70)

multi_ids = (
    sizes[sizes >= 3]
    .sort_values(ascending=False)
    .head(5)
    .index
)

for i, conversation_id in enumerate(multi_ids, 1):

    conversation = result[
        result["conversation_id"] == conversation_id
    ].sort_values("created_at")

    print()
    print(f"--- Conversation {i} ({len(conversation)} tweets) ---")

    for _, row in conversation.iterrows():

        speaker = "CUSTOMER" if row["inbound"] else "APPLE"

        text = str(row["text"]).replace("\n", " ")

        print(f"[{speaker}] {text}")

print()
print("Done.")
