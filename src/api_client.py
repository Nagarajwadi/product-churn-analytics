import requests


BASE_URL = "https://dummyjson.com"


def fetch_users(limit=10, skip=0):
    """Fetch one page of users from DummyJSON API."""
    url = f"{BASE_URL}/users"

    response = requests.get(
        url,
        params={"limit": limit, "skip": skip},
        headers={
            "User-Agent": "product-churn-analytics/1.0",
            "Accept": "application/json",
        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if "users" not in data:
        raise ValueError("API response does not contain 'users'")

    return data


def fetch_all_users(page_size=50):
    """Fetch all users using API pagination."""
    all_users = []
    skip = 0

    while True:
        data = fetch_users(limit=page_size, skip=skip)
        users = data["users"]

        all_users.extend(users)

        if len(all_users) >= data["total"] or not users:
            break

        skip += page_size

    return all_users


def transform_users(users):
    """Keep only fields needed for analytics."""
    transformed = []

    for user in users:
        transformed.append(
            {
                "external_user_id": user["id"],
                "age": user["age"],
                "gender": user["gender"],
                "role": user["role"],
                "department": user["company"]["department"],
                "job_title": user["company"]["title"],
                "country": user["address"]["country"],
            }
        )

    return transformed


if __name__ == "__main__":
    raw_users = fetch_all_users(page_size=50)
    users = transform_users(raw_users)

    print(f"Fetched users: {len(users)}")
    print("First 3 transformed records:")

    for user in users[:3]:
        print(user)
