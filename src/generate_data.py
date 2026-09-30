import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_SEED = 42
NUM_USERS = 5000

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------
# Product configuration
# --------------------------------------------------

COUNTRIES = [
    "India",
    "United States",
    "United Kingdom",
    "Canada",
    "Germany",
    "Australia",
]

DEVICES = [
    "web",
    "mobile",
    "tablet",
]

PLANS = [
    "free",
    "basic",
    "premium",
]


# --------------------------------------------------
# Date configuration
# --------------------------------------------------

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 6, 30)


# --------------------------------------------------
# Generate users
# --------------------------------------------------

def generate_users(num_users):
    users = []

    for i in range(1, num_users + 1):
        user = {
            "user_id": f"U{i:05d}",
            "country": random.choice(COUNTRIES),
            "device": random.choice(DEVICES),
            "plan": random.choices(
                PLANS,
                weights=[0.60, 0.25, 0.15],
                k=1
            )[0],
        }

        users.append(user)

    return pd.DataFrame(users)


# --------------------------------------------------
# Choose user behavior
# --------------------------------------------------

def choose_user_behavior():
    return random.choices(
        ["high", "medium", "low"],
        weights=[0.20, 0.50, 0.30],
        k=1
    )[0]


# --------------------------------------------------
# Generate user events
# --------------------------------------------------

def generate_user_events(user):

    events = []

    behavior = choose_user_behavior()

    signup_time = START_DATE + timedelta(
        days=random.randint(
            0,
            (END_DATE - START_DATE).days
        )
    )

    session_id = 1

    # --------------------------------------------------
    # Signup
    # --------------------------------------------------

    events.append({
        "user_id": user["user_id"],
        "timestamp": signup_time,
        "event_name": "signup",
        "device": user["device"],
        "country": user["country"],
        "plan": user["plan"],
        "session_id": f"{user['user_id']}_S{session_id:04d}"
    })

    # --------------------------------------------------
    # Engagement level
    # --------------------------------------------------

    if behavior == "high":
        active_days = random.randint(20, 60)
        purchase_probability = 0.30
        subscription_probability = 0.70
        cancellation_probability = 0.10

    elif behavior == "medium":
        active_days = random.randint(7, 20)
        purchase_probability = 0.15
        subscription_probability = 0.40
        cancellation_probability = 0.25

    else:
        active_days = random.randint(1, 6)
        purchase_probability = 0.05
        subscription_probability = 0.15
        cancellation_probability = 0.50

    purchased = False
    subscribed = False
    subscription_time = None

    # --------------------------------------------------
    # Generate activity
    # --------------------------------------------------

    for day in range(active_days):

        event_date = signup_time + timedelta(days=day)

        if event_date > END_DATE:
            break

        session_id += 1

        current_session = (
            f"{user['user_id']}_S{session_id:04d}"
        )

        # Login
        events.append({
            "user_id": user["user_id"],
            "timestamp": event_date,
            "event_name": "login",
            "device": user["device"],
            "country": user["country"],
            "plan": user["plan"],
            "session_id": current_session
        })

        # Product view
        events.append({
            "user_id": user["user_id"],
            "timestamp": event_date + timedelta(
                minutes=random.randint(1, 30)
            ),
            "event_name": "view_product",
            "device": user["device"],
            "country": user["country"],
            "plan": user["plan"],
            "session_id": current_session
        })

        # Search
        if random.random() < 0.60:
            events.append({
                "user_id": user["user_id"],
                "timestamp": event_date + timedelta(
                    minutes=random.randint(31, 60)
                ),
                "event_name": "search",
                "device": user["device"],
                "country": user["country"],
                "plan": user["plan"],
                "session_id": current_session
            })

        # Add to cart
        if random.random() < 0.35:
            events.append({
                "user_id": user["user_id"],
                "timestamp": event_date + timedelta(
                    minutes=random.randint(61, 90)
                ),
                "event_name": "add_to_cart",
                "device": user["device"],
                "country": user["country"],
                "plan": user["plan"],
                "session_id": current_session
            })

        # --------------------------------------------------
        # Purchase
        # --------------------------------------------------

        if not purchased and random.random() < purchase_probability:

            purchase_time = event_date + timedelta(
                minutes=random.randint(91, 120)
            )

            events.append({
                "user_id": user["user_id"],
                "timestamp": purchase_time,
                "event_name": "purchase",
                "device": user["device"],
                "country": user["country"],
                "plan": user["plan"],
                "session_id": current_session
            })

            purchased = True

            # --------------------------------------------------
            # Subscription after purchase
            # --------------------------------------------------

            if random.random() < subscription_probability:

                subscription_time = purchase_time + timedelta(
                    minutes=random.randint(5, 30)
                )

                if subscription_time <= END_DATE:

                    events.append({
                        "user_id": user["user_id"],
                        "timestamp": subscription_time,
                        "event_name": "subscription",
                        "device": user["device"],
                        "country": user["country"],
                        "plan": user["plan"],
                        "session_id": current_session
                    })

                    subscribed = True

    # --------------------------------------------------
    # Cancellation / churn
    # --------------------------------------------------

    if subscribed and subscription_time is not None:

        if random.random() < cancellation_probability:

            # Cancellation happens AFTER subscription
            cancellation_day = random.randint(30, 90)

            cancellation_time = (
                subscription_time
                + timedelta(days=cancellation_day)
            )

            if cancellation_time <= END_DATE:

                events.append({
                    "user_id": user["user_id"],
                    "timestamp": cancellation_time,
                    "event_name": "cancel_subscription",
                    "device": user["device"],
                    "country": user["country"],
                    "plan": user["plan"],
                    "session_id": f"{user['user_id']}_S{session_id:04d}"
                })

    return events


# --------------------------------------------------
# Generate full dataset
# --------------------------------------------------

if __name__ == "__main__":

    print("Generating users...")

    users = generate_users(NUM_USERS)

    print(f"Generated {len(users)} users.")

    print()
    print("Generating user events...")

    all_events = []

    for index, (_, user) in enumerate(
        users.iterrows(),
        start=1
    ):

        user_events = generate_user_events(
            user.to_dict()
        )

        all_events.extend(user_events)

        if index % 500 == 0:
            print(
                f"Processed {index}/{NUM_USERS} users..."
            )

    events_df = pd.DataFrame(all_events)

    # Sort events
    events_df = events_df.sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)

    # Save dataset
    output_path = "data/raw/raw_events.csv"

    events_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("===================================")
    print("DATA GENERATION COMPLETE")
    print("===================================")

    print(f"Users:  {len(users):,}")
    print(f"Events: {len(events_df):,}")

    print()
    print("Event types:")

    print(
        events_df["event_name"]
        .value_counts()
    )

    print()
    print(f"Saved to: {output_path}")