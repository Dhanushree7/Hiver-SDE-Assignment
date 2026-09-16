import pandas as pd
import re

INPUT = "data/processed/applesupport_clean_text.csv"

df = pd.read_csv(INPUT)
text = df["clean_customer_text"].fillna("").str.lower()

intent_patterns = {
    "ios_update_issues": [
        r"\bios\b", r"\bupdate\b", r"\bupdated\b",
        r"\bupdating\b", r"\bupgrade\b", r"\bupgraded\b"
    ],

    "battery_power": [
        r"\bbattery\b", r"\bcharging\b", r"\bcharger\b",
        r"\boverheat\b", r"\bhot\b", r"\bcharge\b"
    ],

    "device_performance": [
        r"\bfreez", r"\blag\b", r"\bslow\b",
        r"\bcrash", r"\bhanging\b", r"\bunresponsive\b",
        r"\brestart\b", r"\breboot\b"
    ],

    "keyboard_input": [
        r"\bkeyboard\b", r"\bautocorrect\b",
        r"\btyping\b", r"\btype\b",
        r"\bquestion mark\b", r"\bquestionmark\b"
    ],

    "apple_id_account": [
        r"\bapple id\b", r"\bicloud\b", r"\bpassword\b",
        r"\blocked\b", r"\baccount\b", r"\bverification\b",
        r"\bsecurity\b"
    ],

    "apps_services": [
        r"\bapp store\b", r"\bitunes\b", r"\bapple music\b",
        r"\bmusic app\b", r"\bmail app\b", r"\bsiri\b",
        r"\bapp\b", r"\bapps\b"
    ],

    "connectivity": [
        r"\bwi[- ]?fi\b", r"\bbluetooth\b", r"\bsignal\b",
        r"\bsim\b", r"\bcellular\b", r"\bnetwork\b",
        r"\bmobile data\b", r"\bfacetime\b"
    ],

    "photos_data": [
        r"\bphoto", r"\bpictures?\b", r"\bcontacts?\b",
        r"\bdata\b", r"\bbackup\b", r"\bicloud photos\b"
    ],

    "purchases_orders_refunds": [
        r"\border\b", r"\bordered\b", r"\brefund\b",
        r"\bpurchase\b", r"\bbought\b", r"\bsubscription\b",
        r"\bcheckout\b", r"\bpreorder\b", r"\bpre-order\b"
    ],

    "hardware_accessories": [
        r"\bcharger\b", r"\bheadphones?\b", r"\bearpods?\b",
        r"\bcable\b", r"\btouch screen\b", r"\btouchscreen\b",
        r"\bscreen\b", r"\bcamera\b", r"\bkeyboard\b"
    ]
}

print("=" * 70)
print("CANDIDATE INTENT COVERAGE")
print("=" * 70)

results = []

for intent, patterns in intent_patterns.items():

    mask = pd.Series(False, index=df.index)

    for pattern in patterns:
        mask = mask | text.str.contains(
            pattern,
            regex=True,
            na=False
        )

    count = int(mask.sum())
    percentage = count / len(df) * 100

    results.append((intent, count, percentage))

for intent, count, percentage in sorted(
    results,
    key=lambda x: x[1],
    reverse=True
):
    print(f"{intent:25s} {count:8,d}  ({percentage:5.1f}%)")

print()
print("=" * 70)
print(f"Total messages: {len(df):,}")
print("=" * 70)

print()
print("Note: these are keyword coverage estimates, NOT final labels.")
