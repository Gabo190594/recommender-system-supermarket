import numpy as np
import pandas as pd
import os

np.random.seed(42)

# --------------------------
# CONFIG
# --------------------------
N_USERS = 500
N_PRODUCTS = 800
N_INTERACTIONS = 15000
N_RATINGS = 4000
N_SOCIAL_LINKS = 1000

# --------------------------
# 1. USERS
# --------------------------
users = pd.DataFrame({
    "user_id": range(1, N_USERS + 1),
    "age": np.random.randint(18, 65, N_USERS),
    "segment": np.random.choice(
        ["familia", "soltero", "fitness", "gourmet"], N_USERS
    )
})

# --------------------------
# 2. PRODUCTS
# --------------------------
categories = ["frutas", "verduras", "bebidas", "snacks", "lacteos", "carnes", "limpieza", "hogar"]

products = pd.DataFrame({
    "product_id": range(1, N_PRODUCTS + 1),
    "category": np.random.choice(categories, N_PRODUCTS),
    "price": np.round(np.random.uniform(1, 100, N_PRODUCTS), 2)
})

# --------------------------
# 3. INTERACTIONS (implícitas)
# --------------------------
interactions = pd.DataFrame({
    "user_id": np.random.choice(users.user_id, N_INTERACTIONS),
    "product_id": np.random.choice(products.product_id, N_INTERACTIONS),
    "event_type": np.random.choice(
        ["view", "cart", "purchase"],
        N_INTERACTIONS,
        p=[0.65, 0.2, 0.15]
    ),
    "timestamp": pd.to_datetime("2025-01-01") +
                 pd.to_timedelta(np.random.randint(0, 180, N_INTERACTIONS), unit='d')
})

event_score = {"view": 1, "cart": 3, "purchase": 5}
interactions["implicit_score"] = interactions["event_type"].map(event_score)

# --------------------------
# 4. RATINGS (explícito)
# --------------------------
ratings = pd.DataFrame({
    "user_id": np.random.choice(users.user_id, N_RATINGS),
    "product_id": np.random.choice(products.product_id, N_RATINGS),
    "rating": np.random.randint(1, 6, N_RATINGS)
})

# --------------------------
# 5. SOCIAL LINKS
# --------------------------
social_links = pd.DataFrame({
    "user_id": np.random.choice(users.user_id, N_SOCIAL_LINKS),
    "friend_id": np.random.choice(users.user_id, N_SOCIAL_LINKS)
})

social_links = social_links[social_links["user_id"] != social_links["friend_id"]]

# --------------------------
# 6. INFLUENCIA SOCIAL
# --------------------------
def apply_social_influence(interactions, social_links):
    influenced = interactions.copy()

    sample_idx = influenced.sample(frac=0.15, random_state=42).index

    for idx in sample_idx:
        user = influenced.at[idx, "user_id"]
        friends = social_links[social_links["user_id"] == user]["friend_id"].values

        if len(friends) > 0:
            friend_purchases = interactions[
                (interactions["user_id"].isin(friends)) &
                (interactions["event_type"] == "purchase")
            ]

            if not friend_purchases.empty:
                influenced.at[idx, "product_id"] = friend_purchases.sample(1)["product_id"].values[0]

    return influenced

interactions = apply_social_influence(interactions, social_links)

# --------------------------
# 7. GUARDAR DATA
# --------------------------
os.makedirs("data/raw", exist_ok=True)

users.to_csv("data/raw/users.csv", index=False)
products.to_csv("data/raw/products.csv", index=False)
interactions.to_csv("data/raw/interactions.csv", index=False)
ratings.to_csv("data/raw/ratings.csv", index=False)
social_links.to_csv("data/raw/social_links.csv", index=False)

print("✅ Data sintética generada en data/raw/")
