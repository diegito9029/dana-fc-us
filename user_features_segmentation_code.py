#!/usr/bin/env python
# coding: utf-8

# # Initial Data Analysis (IDA)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)


# ### Data Importing

# In[2]:


df = pd.read_csv('./data/transactions.csv')


# ### Data Checking

# In[3]:


print(df)
# display(df)


# In[4]:


print("Rows:", df.shape[0], "\n")

print("Columns:", df.shape[1])
for i in df.columns:
    print(f'- {i} ({df[i].dtype})')


# In[5]:


display(df.info())


# In[6]:


display(df.isnull().sum())


# ### Data Description

# In[7]:


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric Columns:" , numeric_cols)
# display(df[["transaction_amount", "promo_amount"]].describe().T)


# In[8]:


categorical_cols = [
    "user_id",
    "merchant_id",
    "merchant_name",
    "merchant_category_id",
    "payment_method",
    "user_agent",
    "loyalty_program",
    "discount_applied",
    "transaction_status",
    "is_refunded",
]

print("Categorical Columns:", categorical_cols)
for col in categorical_cols:
    if col in df.columns:
        pass
        # display(df[col].value_counts())


# ### Data Parsing

# #### transaction_date

# In[9]:


# Parse transaction_date and Add Temporal Features

# Convert to datetime
df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

# Check for issues
n_missing_dates = df["transaction_date"].isna().sum()
print(f"Number of Issues: {n_missing_dates}")

# Add Temporal Features
df["date"] = df["transaction_date"].dt.date
# df["hour"] = df["transaction_date"].dt.hour
df["day_of_week"] = df["transaction_date"].dt.dayofweek  # Monday=0, Sunday=6
df["is_weekend"] = df["day_of_week"].isin([5, 6]).map({True: "yes", False: "no"})
df["month"] = df["transaction_date"].dt.month

iso_calendar = df["transaction_date"].dt.isocalendar()
df["week_of_year"] = iso_calendar["week"]


# In[10]:


print("\ntransaction_date and Temporal Features:")
display(df[["transaction_date", "date", "day_of_week", "is_weekend", "month", "week_of_year"]].head())


# #### geo_location

# In[11]:


# Parse geo_location and Add Latitude and Longitude

# Convert to lat, long
def parse_geo_location(geo_str):
    if pd.isna(geo_str):
        return np.nan, np.nan
    try:
        parts = str(geo_str).split(",")
        if len(parts) != 2:
            return np.nan, np.nan
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        return lat, lon
    except Exception:
        return np.nan, np.nan

latitudes = []
longitudes = []

for val in df["geo_location"]:
    lat, lon = parse_geo_location(val)
    latitudes.append(lat)
    longitudes.append(lon)

df["latitude"] = latitudes
df["longitude"] = longitudes

print("Number of Issues in Latitude:", df["latitude"].isna().sum())
print("Number of Issues in Longitude:", df["longitude"].isna().sum())


# In[12]:


print("\ngeo_location and Latitude and Longitude:")
display(df[["geo_location", "latitude", "longitude"]].head())


# In[13]:


plt.figure(figsize=(10, 10))
plt.scatter(df["longitude"], df["latitude"], alpha=0.1, s=2)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Locations")
plt.show()


# In[14]:


# Cluster geo_location into 9 and Find Nearest City

# Cluster geo_location and find the centroids
coords = df[["latitude", "longitude"]]

kmeans = KMeans(n_clusters=9, random_state=42)
df["cluster"] = kmeans.fit_predict(coords)

cluster_centers = df.groupby("cluster")[["latitude", "longitude"]].mean()
print(cluster_centers)


# In[15]:


# Convert centroids to city
nearest_city_map = {
    0: "Depok",
    1: "Malang",
    2: "Yogyakarta",
    3: "Semarang",
    4: "Surabaya",
    5: "Jakarta",
    6: "Tangerang",
    7: "Bogor",
    8: "Bekasi"
}

df["nearest_major_city"] = df["cluster"].map(nearest_city_map)


# In[16]:


plt.figure(figsize=(10, 10))
plt.scatter(df["longitude"], df["latitude"], c=df["cluster"], s=5, cmap="tab10")
plt.scatter(cluster_centers["longitude"], cluster_centers["latitude"],
            s=200, c="black", marker="x")

for cluster_id, row in cluster_centers.iterrows():
    city = nearest_city_map[cluster_id]
    plt.text(
        row["longitude"] + 0.02,
        row["latitude"] + 0.02,
        city,
        fontsize=10,
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Locations w/ Clusters & City")
plt.show()


# ### Data Standarization

# In[17]:


# Standardize Categorical Columns & Map Boolean-Like Flags

# Columns to strip/normalize text casing
cols_to_clean = [
    "transaction_id",
    "geo_location",
    "date",
]

for col in cols_to_clean:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()


# Map yes/no to 0/1 for certain flags
yes_no_map = {"yes": 1, "no": 0}

for col in ["loyalty_program", "discount_applied", "transaction_notes", "is_refunded", "is_weekend"]:
    df[col] = df[col].map(yes_no_map)
        

# Map completed/failed to 0/1 for transaction_status
completed_failed_map = {"completed": 1, "failed": 0}
df["transaction_status"] = df["transaction_status"].map(completed_failed_map)
        

# Convert columns to categorical dtype
categorical_to_convert = [
    "user_id",
    "merchant_id",
    "merchant_name",
    "merchant_category_id",
    "payment_method",
    "user_agent",
    "loyalty_program",
    "discount_applied",
    "transaction_notes",
    "merchant_rating",
    "transaction_status",
    "is_refunded",
    "day_of_week",
    "is_weekend",
    "month",
    "week_of_year",
    "cluster",
    "nearest_major_city",
]

for col in categorical_to_convert:
    if col in df.columns:
        df[col] = df[col].astype("category")


# ### Data Checking

# In[18]:


# print(df)
# display(df)


# In[19]:


display(df.info())


# In[20]:


# print(df.isnull().sum())
display(df.isnull().sum())


# ### Data Description

# In[21]:


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric Columns:" , numeric_cols)
# print(df[numeric_cols].describe().T)
# display(df[numeric_cols].describe().T)


# In[22]:


categorical_cols = df.select_dtypes(include=["category"]).columns.tolist()

print("Categorical Columns:", categorical_cols)
for col in categorical_cols:
    pass
    # print(df[col].value_counts())
    # display(df[col].value_counts())


# ### Data Visualization

# In[23]:


# transaction_amount

plt.figure(figsize=(10, 10))
plt.hist(df["transaction_amount"], bins=50)
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.title("Transaction Amount")
plt.tight_layout()
plt.show()


# In[24]:


# promo_amount

plt.figure(figsize=(10, 10))
plt.hist(df["promo_amount"], bins=50)
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.title("Promo Amount")
plt.tight_layout()
plt.show()


# ### Data Exporting

# In[25]:


# df.to_csv('./data/transactions_ida.csv', index=False)
df.to_parquet("./data/transactions_ida.parquet", index=False)


# In[26]:


get_ipython().run_line_magic('reset', '-f')


# # Exploratory Data Analysis (EDA)

# ## Feature Engineering: User-Level Features

# In[27]:


get_ipython().run_line_magic('reset', '-f')


# In[28]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)


# ### Data Importing

# In[29]:


# df = pd.read_csv('./data/transactions_ida.csv')
df = pd.read_parquet('./data/transactions_ida.parquet')


# In[30]:


display(df.info())


# ### Data Grouping

# In[31]:


# Group by user_id

user_group = df.groupby("user_id")

# Calculate number of transactions per user
user_base = user_group["transaction_id"].size().to_frame("n_transactions")

print("Number of Users:", user_base.shape[0])
print(user_base)


# ### Feature Creation: Spend / Volume + Completion / Refund

# In[32]:


# Transaction Spend / Volume

user_spend = user_group["transaction_amount"].agg(
    total_spend="sum",
    avg_transaction_amount="mean",
    median_transaction_amount="median",
    std_transaction_amount="std",
    max_transaction_amount="max",
    min_transaction_amount="min",
)

# Coefficient of variation (handle NaNs / inf)
user_spend["cv_transaction_amount"] = (
    user_spend["std_transaction_amount"] / user_spend["avg_transaction_amount"]
)
user_spend["cv_transaction_amount"] = user_spend["cv_transaction_amount"].replace(
    [np.inf, -np.inf], np.nan
)
user_spend["cv_transaction_amount"] = user_spend["cv_transaction_amount"].fillna(0)

# Log-transformed spend
# user_spend["log_total_spend"] = np.log1p(user_spend["total_spend"])
# user_spend["log_avg_transaction_amount"] = np.log1p(user_spend["avg_transaction_amount"])


# In[33]:


print("User-Level Spend / Volume:")
display(user_spend.head())


# In[34]:


# Transaction Completion / Refund

# transaction_status and is_refunded are categorical with values {0,1}
df["transaction_status_num"] = df["transaction_status"].astype(int)
df["is_refunded_num"] = df["is_refunded"].astype(int)

user_status = user_group["transaction_status_num"].agg(
    n_completed="sum",
    completion_rate="mean"  # proportion of completed transactions
)

user_refund = user_group["is_refunded_num"].agg(
    n_refunded="sum",
    refund_rate="mean"  # proportion of refunded transactions
)


# In[35]:


print("User-Level Completion:")
display(user_status.head())


# In[36]:


print("User-Level Refund:")
display(user_refund.head())


# ### Feature Creation: Temporal

# In[37]:


# Transaction Timeline + Active Day + Recency

user_temporal = user_group["transaction_date"].agg(
    first_txn_date="min",
    last_txn_date="max",
)

# Activity span in days (inclusive)
user_temporal["activity_span_days"] = (
    user_temporal["last_txn_date"] - user_temporal["first_txn_date"]
).dt.days + 1

# Number of unique active days (dates)
user_active_days = user_group["date"].nunique().to_frame("n_active_days")
user_temporal = user_temporal.join(user_active_days)

# Average transactions per active day
user_temporal["avg_txn_per_active_day"] = (
    user_base["n_transactions"] / user_temporal["n_active_days"]
)

# Recency: days since last transaction relative to max date in dataset
max_date = df["transaction_date"].max()
user_temporal["recency_days"] = (max_date - user_temporal["last_txn_date"]).dt.days


# In[38]:


print("User-Level Timeline + Active Day + Recency:")
display(user_temporal.head())


# In[39]:


# Transaction Weekend vs Weekday Behavior

df["is_weekend_num"] = df["is_weekend"].astype(int)

user_weekend = user_group["is_weekend_num"].mean().to_frame("weekend_txn_fraction")
user_weekend["weekday_txn_fraction"] = 1 - user_weekend["weekend_txn_fraction"]


# In[40]:


print("User-Level Weekend vs Weekday Behavior:")
display(user_weekend.head())


# In[41]:


# Transaction Day-of-Week Distribution

user_dow = pd.crosstab(df["user_id"], df["day_of_week"], normalize="index")
user_dow.columns = [f"dow_{int(col)}_share" for col in user_dow.columns]


# In[42]:


print("User-Level Day-of-Week Distribution:")
display(user_dow.head())


# In[43]:


# Transaction Month-of-Year Distribution

user_month = pd.crosstab(df["user_id"], df["month"], normalize="index")
user_month.columns = [f"month_{int(col)}_share" for col in user_month.columns]


# In[44]:


print("User-Level Month-of-Year Distribution:")
display(user_month.head())


# ### Feature Creation: Merchant

# In[45]:


# Transaction Category & Merchant Diversity

user_cat_basic = pd.DataFrame(index=user_base.index)
user_cat_basic["n_distinct_categories"] = user_group["merchant_category_id"].nunique()
user_cat_basic["n_distinct_merchants"] = user_group["merchant_id"].nunique()


# In[46]:


print("User-Level Category & Merchant Diversity:")
display(user_cat_basic.head())


# In[47]:


# Transaction Category Entropy

def compute_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return -np.sum(probs * np.log2(probs + 1e-12))

cat_counts = (
    df.groupby(["user_id", "merchant_category_id"])
        .size()
        .to_frame("count")
        .reset_index()
)

user_category_entropy = (
    cat_counts.groupby("user_id")["count"]
    .apply(lambda x: compute_entropy(x.values))
    .to_frame("category_entropy")
)


# In[48]:


print("User-Level Category Entropy:")
display(user_category_entropy.head())


# In[49]:


# Transaction Merchant Entropy

merch_counts = (
    df.groupby(["user_id", "merchant_id"])
        .size()
        .to_frame("count")
        .reset_index()
)

user_merchant_entropy = (
    merch_counts.groupby("user_id")["count"]
    .apply(lambda x: compute_entropy(x.values))
    .to_frame("merchant_entropy")
)


# In[50]:


print("User-Level Merchant Entropy:")
display(user_merchant_entropy.head())


# In[ ]:


# Transaction Top Merchant Spend Share

user_merchant_spend = (
    df.groupby(["user_id", "merchant_id"])["transaction_amount"]
        .sum()
        .reset_index()
)

# For each user, find the merchant with highest total spend
idx = user_merchant_spend.groupby("user_id")["transaction_amount"].idxmax()
top_merchant = user_merchant_spend.loc[idx].set_index("user_id")

# Share of spend at top merchant
top_merchant["top_merchant_spend_share"] = (
    top_merchant["transaction_amount"] / user_spend["total_spend"]
)

user_top_merchant_share = top_merchant[["top_merchant_spend_share"]]


# In[ ]:


print("User-Level Top Merchant Spend Share:")
display(user_top_merchant_share.head())


# In[ ]:


# Transaction Category Count Share for Top 10 MCCs
# Transaction Category Spend Share for Top 10 MCCs

# Identify top 10 merchant_category_id by overall transaction count
top_mcc_ids = df["merchant_category_id"].value_counts().head(10).index

df_top_mcc = df[df["merchant_category_id"].isin(top_mcc_ids)].copy()

# Count share: fraction of transactions in each of top MCCs
user_cat_count_share = pd.crosstab(
    df_top_mcc["user_id"],
    df_top_mcc["merchant_category_id"],
    normalize="index",
)
user_cat_count_share.columns = [
    f"cat_{int(col)}_count_share" for col in user_cat_count_share.columns
]

# Spend share: fraction of total spend in each of top MCCs
user_cat_spend = df_top_mcc.pivot_table(
    index="user_id",
    columns="merchant_category_id",
    values="transaction_amount",
    aggfunc="sum",
)

user_cat_spend = user_cat_spend.div(user_spend["total_spend"], axis=0)
user_cat_spend = user_cat_spend.fillna(0)
user_cat_spend.columns = [
    f"cat_{int(col)}_spend_share" for col in user_cat_spend.columns
]


# In[ ]:


print("User-Level Category Count Share for Top 10 MCCs:")
display(user_cat_count_share.head())

print("User-Level Category Spend Share for Top 10 MCCs:")
display(user_cat_spend.head())


# ### Feature Creation: Location

# In[ ]:


# Transaction Distinct Locations

# Number of distinct lat/lon pairs per user
unique_locs = df[["user_id", "latitude", "longitude"]].drop_duplicates()

user_geo_basic = pd.DataFrame(index=user_base.index)
user_geo_basic["n_distinct_locations"] = (
    unique_locs.groupby("user_id").size().reindex(user_base.index, fill_value=0)
)


# In[ ]:


print("User-Level Distinct Locations:")
display(user_geo_basic.head())


# In[ ]:


# Transaction Geo Centroids

user_geo_centroid = df.groupby("user_id")[["latitude", "longitude"]].mean()
user_geo_centroid = user_geo_centroid.rename(
    columns={"latitude": "centroid_lat", "longitude": "centroid_lon"}
)


# In[ ]:


print("User-Level Geo Centroids:")
display(user_geo_centroid.head())


# In[ ]:


# Transaction Geo Dispersion

def haversine_array(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance for arrays of lat/lon in degrees.
    Returns distance in kilometers.
    """
    R = 6371.0  # Earth radius in km
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df_geo = df[["user_id", "latitude", "longitude"]].merge(
    user_geo_centroid,
    left_on="user_id",
    right_index=True,
    how="left",
)

df_geo["dist_to_centroid_km"] = haversine_array(
    df_geo["latitude"],
    df_geo["longitude"],
    df_geo["centroid_lat"],
    df_geo["centroid_lon"],
)

user_geo_dispersion = (
    df_geo.groupby("user_id")["dist_to_centroid_km"]
    .mean()
    .to_frame("avg_distance_from_centroid_km")
)


# In[ ]:


print("User-Level Geo Dispersion:")
display(user_geo_dispersion.head())


# In[ ]:


# Transaction City Behaviour

# Number of distinct nearest_major_city per user
user_city = df.groupby("user_id")["nearest_major_city"].nunique().to_frame("n_cities")

# Primary city: most frequent nearest_major_city
user_primary_city = (
    df.groupby("user_id")["nearest_major_city"]
    .agg(lambda x: x.value_counts().idxmax())
    .to_frame("primary_city")
)

# City share features: fraction of transactions in each city
user_city_share = pd.crosstab(
    df["user_id"], df["nearest_major_city"], normalize="index"
)
user_city_share.columns = [f"city_{str(col)}_share" for col in user_city_share.columns]


# In[ ]:


print("User-Level City Behaviour:")
display(user_city.head())
display(user_primary_city.head())
display(user_city_share.head())


# ### Feature Creation: Payment + Device

# In[ ]:


# Transaction Payment

# Indicator for credit card transactions
df["is_credit_card"] = (df["payment_method"] == "credit_card").astype(int)

user_payment = user_group["is_credit_card"].agg(
    n_credit_card="sum",
    credit_card_txn_fraction="mean",
)

# Number of balance (non-credit) transactions and their fraction
user_payment["n_balance"] = (
    user_base["n_transactions"] - user_payment["n_credit_card"]
)
user_payment["balance_txn_fraction"] = (
    user_payment["n_balance"] / user_base["n_transactions"]
)


# In[ ]:


print("User-Level Payment:")
display(user_payment.head())


# In[ ]:


# Transaction Device

def get_device_family(ua):
    ua = str(ua).lower()
    if "android" in ua:
        return "android"
    if "iphone" in ua:
        return "iphone"
    if "ipad" in ua:
        return "ipad"
    return "other"

df["device_family"] = df["user_agent"].astype(str).apply(get_device_family)
df["device_family"] = df["device_family"].astype("category")

# Number of distinct device families per user
# user_device_counts = (
#     df.groupby("user_id")["device_family"]
#     .nunique()
#     .to_frame("n_device_families")
# )

# Primary device family (mode)
user_primary_device = (
    df.groupby("user_id")["device_family"]
    .agg(lambda x: x.value_counts().idxmax())
    .to_frame("primary_device_family")
)

# Device family share features: fraction of transactions from each device type
# user_device_share = pd.crosstab(
#     df["user_id"], df["device_family"], normalize="index"
# )
# user_device_share.columns = [
#     f"device_{str(col)}_share" for col in user_device_share.columns
# ]


# In[ ]:


print("User-Level Device:")
# display(user_device_counts.head())
display(user_primary_device.head())
# display(user_device_share.head())


# ### Feature Creation: Loyalty Program + Discount + Promo

# In[ ]:


# Transaction Loyalty Program + Discount

df["loyalty_program_num"] = df["loyalty_program"].astype(int)
df["discount_applied_num"] = df["discount_applied"].astype(int)
df["transaction_notes_num"] = df["transaction_notes"].astype(int)

# Loyalty metrics
user_loyalty = user_group["loyalty_program_num"].agg(
    loyalty_txn_count="sum",
    loyalty_txn_fraction="mean",
)
user_loyalty["is_loyalty_member"] = (
    user_loyalty["loyalty_txn_count"] > 0
).astype(int)

# Discount metrics
user_discount = user_group["discount_applied_num"].agg(
    discount_txn_count="sum",
    discount_txn_fraction="mean",
)

# # Notes presence: fraction of transactions with notes
# user_notes = user_group["transaction_notes_num"].mean().to_frame(
#     "notes_txn_fraction"
# )


# In[ ]:


print("User-Level Loyalty Program:")
display(user_loyalty.head())

print("User-Level Discount:")
display(user_discount.head())


# In[ ]:


# Transaction Promo

user_promo = user_group["promo_amount"].agg(
    total_promo_amount="sum",
    avg_promo_amount="mean",
)

# Promo to spend ratio
user_promo["promo_to_spend_ratio"] = (
    user_promo["total_promo_amount"] / user_spend["total_spend"]
).fillna(0)


# In[ ]:


print("User-Level Promo:")
display(user_promo.head())


# ### Feature Creation: Ratings

# In[ ]:


# Transaction Ratings

df["merchant_rating_num"] = df["merchant_rating"].astype(int)

user_rating_basic = user_group["merchant_rating_num"].agg(
    avg_rating="mean",
    rating_std="std",
    min_rating="min",
    max_rating="max",
)

# Count of low ratings (<=2) and high ratings (>=4)
low_rating_mask = df["merchant_rating_num"] <= 2
high_rating_mask = df["merchant_rating_num"] >= 4

user_low_ratings = (
    df[low_rating_mask].groupby("user_id").size().to_frame("n_low_ratings")
)
user_high_ratings = (
    df[high_rating_mask].groupby("user_id").size().to_frame("n_high_ratings")
)

# Reindex to include users with 0 low/high ratings
user_low_ratings = user_low_ratings.reindex(user_base.index, fill_value=0)
user_high_ratings = user_high_ratings.reindex(user_base.index, fill_value=0)

# Combine rating features
user_rating = user_rating_basic.join(user_low_ratings, how="left").join(
    user_high_ratings, how="left"
)


# In[ ]:


print("User-Level Ratings:")
display(user_rating.head())


# ### Feature Selection

# In[ ]:


user_features = user_base.join(user_spend, how="left")

# Spend / Volume + Completion / Refund
user_features = user_features.join(user_status, how="left")
user_features = user_features.join(user_refund, how="left")

# Temporal
user_features = user_features.join(user_temporal, how="left")
user_features = user_features.join(user_weekend, how="left")
user_features = user_features.join(user_dow, how="left")
user_features = user_features.join(user_month, how="left")

# Merchant
user_features = user_features.join(user_cat_basic, how="left")
user_features = user_features.join(user_category_entropy, how="left")
user_features = user_features.join(user_merchant_entropy, how="left")
user_features = user_features.join(user_top_merchant_share, how="left")
user_features = user_features.join(user_cat_count_share, how="left")
user_features = user_features.join(user_cat_spend, how="left")

# Location
user_geo_all = user_geo_basic.join(user_geo_centroid, how="left")
user_geo_all = user_geo_all.join(user_geo_dispersion, how="left")
user_geo_all = user_geo_all.join(user_city, how="left")
user_geo_all = user_geo_all.join(user_primary_city, how="left")
user_geo_all = user_geo_all.join(user_city_share, how="left")
user_features = user_features.join(user_geo_all, how="left")

# Payment + Device
# user_device_all = user_device_counts.join(user_primary_device, how="left")
# user_device_all = user_device_all.join(user_device_share, how="left")
user_features = user_features.join(user_primary_device, how="left")

user_features = user_features.join(user_payment, how="left")

# Loyalty Program + Discount + Promo
user_features = user_features.join(user_loyalty, how="left")
user_features = user_features.join(user_discount, how="left")
# user_features = user_features.join(user_notes, how="left")
user_features = user_features.join(user_promo, how="left")

# Ratings
user_features = user_features.join(user_rating, how="left")


# ### Data Checking

# In[ ]:


print("User-Level Features:", user_features.shape)
print(user_features.head())
# display(user_features.head())


# In[ ]:


display(user_features.info())


# In[ ]:


print(user_features.isnull().sum())
# display(user_features.isnull().sum())


# ### Data Visualization

# In[ ]:


# Distribution of User Transactions

plt.figure(figsize=(10, 5))
plt.hist(user_features["n_transactions"], bins=50)
plt.xlabel("Transactions per User")
plt.ylabel("Number of Users")
plt.title("Distribution of User Transactions")
plt.tight_layout()
plt.show()


# In[ ]:


# Distribution of User Total Spend

plt.figure(figsize=(10, 5))
plt.hist(user_features["total_spend"], bins=50)
plt.xlabel("Total Spend")
plt.ylabel("Number of Users")
plt.title("Distribution of User Total Spend")
plt.tight_layout()
plt.show()


# In[ ]:


# Total Spend vs Number of Transactions per User

plt.figure(figsize=(10, 5))
plt.scatter(
    user_features["n_transactions"],
    user_features["total_spend"],
    alpha=0.3,
    s=10,
)
plt.xlabel("Number of Transactions")
plt.ylabel("Total Spend")
plt.title("Total Spend vs Number of Transactions per User")
plt.tight_layout()
plt.show()


# In[ ]:


# Distribution of Weekend Transaction Fraction

plt.figure(figsize=(10, 5))
plt.hist(user_features["weekend_txn_fraction"], bins=30)
plt.xlabel("Weekend Transaction Fraction")
plt.ylabel("Number of Users")
plt.title("Distribution of Weekend Transaction Fraction")
plt.tight_layout()
plt.show()


# In[ ]:


# Distribution of Distinct Merchants per User

plt.figure(figsize=(10, 5))
plt.hist(user_features["n_distinct_merchants"].dropna(), bins=50)
plt.xlabel("Number of Distinct Merchants")
plt.ylabel("Number of Users")
plt.title("Distribution of Distinct Merchants per User")
plt.tight_layout()
plt.show()


# ### Data Exporting

# In[ ]:


# user_features.to_csv('./data/transactions_eda_ulf.csv', index=False)
user_features.to_parquet("./data/transactions_eda_ulf.parquet", index=False)


# In[ ]:


get_ipython().run_line_magic('reset', '-f')


# ## Feature Engineering: Derived Features

# In[1]:


get_ipython().run_line_magic('reset', '-f')


# In[2]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)


# ### Data Importing

# In[3]:


# user_features = pd.read_csv('./data/transactions_eda_ulf.csv')
user_features = pd.read_parquet('./data/transactions_eda_ulf.parquet')


# In[4]:


display(user_features.info())


# ### Data Aggregation

# In[5]:


def min_max_scale(series):
    """
    Min-max scale a pandas Series to [0, 1].
    - Handles NaN and inf gracefully.
    - If constant series, returns 0.5 for all.
    """
    s = series.replace([np.inf, -np.inf], np.nan)
    # Fill NaN with mean (or 0 if all NaN)
    if s.isna().all():
        return pd.Series(0.5, index=series.index)
    s = s.fillna(s.mean())
    min_val = s.min()
    max_val = s.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (s - min_val) / (max_val - min_val)


def z_score(series):
    """
    Standard z-score: (x - mean) / std.
    If std is 0, returns 0 for all.
    """
    s = series.replace([np.inf, -np.inf], np.nan)
    s = s.fillna(s.mean())
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (s - s.mean()) / std


# In[6]:


# Category Aggregates

# Ensure missing columns default to 0 (defensive)
for col in [
    "cat_5812_spend_share", "cat_5814_spend_share",  # restaurants & bars
    "cat_5411_spend_share", "cat_5412_spend_share",  # grocery-ish
    "cat_5912_spend_share",                          # pharmacies
    "cat_5942_spend_share",                          # books/stationery
    "cat_5999_spend_share",                          # general retail
    "cat_5541_spend_share",                          # fuel/service stations
]:
    if col not in user_features.columns:
        user_features[col] = 0.0

user_features["restaurant_spend_share"] = (
    user_features["cat_5812_spend_share"] +
    user_features["cat_5814_spend_share"]
)

user_features["grocery_spend_share"] = (
    user_features["cat_5411_spend_share"] +
    user_features["cat_5412_spend_share"]
)

user_features["pharmacy_spend_share"] = user_features["cat_5912_spend_share"]

user_features["general_retail_spend_share"] = (
    user_features["cat_5999_spend_share"] +
    user_features["cat_5942_spend_share"]
)

user_features["fuel_spend_share"] = user_features["cat_5541_spend_share"]

# A "family/household" proxy: groceries + pharmacies
user_features["family_spend_share"] = (
    user_features["grocery_spend_share"] +
    user_features["pharmacy_spend_share"]
)


# In[7]:


print("Category Aggregates:")
display(user_features[[
    "restaurant_spend_share",
    "grocery_spend_share",
    "pharmacy_spend_share",
    "general_retail_spend_share",
    "fuel_spend_share",
    "family_spend_share"
]].head())


# ### 1st Priority

# ### Feature Creation: Big Five Personality Score

# #### Big Five: Openness

# In[8]:


# Big Five: Openness

o_cat_entropy = min_max_scale(user_features["category_entropy"])
o_merch_entropy = min_max_scale(user_features["merchant_entropy"])
o_n_locs = min_max_scale(user_features["n_distinct_locations"])
o_geo_disp = min_max_scale(user_features["avg_distance_from_centroid_km"])
o_n_cities = min_max_scale(user_features["n_cities"])

user_features["big5_openness"] = (
    o_cat_entropy +
    o_merch_entropy +
    o_n_locs +
    o_geo_disp +
    o_n_cities
) / 5.0


# In[9]:


print("Big Five – Openness:")
display(user_features[["big5_openness"]].head())


# In[10]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["big5_openness"], bins=30)
plt.title("Distribution of Big Five - Openness")
plt.xlabel("Score of Openness")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# #### Big Five: Conscientiousness

# In[11]:


# Big Five: Conscientiousness

c_stable_spend = 1 - min_max_scale(user_features["cv_transaction_amount"])
c_low_refund = 1 - min_max_scale(user_features["refund_rate"])
c_completion = min_max_scale(user_features["completion_rate"])
c_loyalty = min_max_scale(user_features["loyalty_txn_fraction"])

user_features["big5_conscientiousness"] = (
    c_stable_spend +
    c_low_refund +
    c_completion +
    c_loyalty
) / 4.0


# In[12]:


print("Big Five – Conscientiousness:")
display(user_features[["big5_conscientiousness"]].head())


# In[13]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["big5_conscientiousness"], bins=30)
plt.title("Distribution of Big Five - Conscientiousness")
plt.xlabel("Score of Conscientiousness")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# #### Big Five: Extraversion

# In[14]:


# Big Five: Extraversion

e_weekend = min_max_scale(user_features["weekend_txn_fraction"])
e_restaurants = min_max_scale(user_features["restaurant_spend_share"])
e_city_diversity = min_max_scale(user_features["n_cities"])

user_features["big5_extraversion"] = (
    e_weekend +
    e_restaurants +
    e_city_diversity
) / 3.0


# In[15]:


print("Big Five – Extraversion:")
display(user_features[["big5_extraversion"]].head())


# In[16]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["big5_extraversion"], bins=30)
plt.title("Distribution of Big Five - Extraversion")
plt.xlabel("Score of Extraversion")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# #### Big Five: Agreeableness

# In[17]:


# Big Five: Agreeableness

low_rating_rate = user_features["n_low_ratings"] / user_features["n_transactions"]

a_high_rating = min_max_scale(user_features["avg_rating"])
a_low_low_rating_rate = 1 - min_max_scale(low_rating_rate)
a_low_refund = 1 - min_max_scale(user_features["refund_rate"])

user_features["big5_agreeableness"] = (
    a_high_rating +
    a_low_low_rating_rate +
    a_low_refund
) / 3.0


# In[18]:


print("Big Five – Agreeableness:")
display(user_features[["big5_agreeableness"]].head())


# In[19]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["big5_agreeableness"], bins=30)
plt.title("Distribution of Big Five - Agreeableness")
plt.xlabel("Score of Agreeableness")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# #### Big Five: Neuroticism

# In[20]:


# Big Five: Neuroticism

n_high_cv = min_max_scale(user_features["cv_transaction_amount"])
n_high_refund = min_max_scale(user_features["refund_rate"])
n_low_avg_rating = 1 - min_max_scale(user_features["avg_rating"])
n_high_low_rating_rate = min_max_scale(low_rating_rate)

user_features["big5_neuroticism"] = (
    n_high_cv +
    n_high_refund +
    n_low_avg_rating +
    n_high_low_rating_rate
) / 4.0


# In[21]:


print("Big Five – Neuroticism:")
display(user_features[["big5_neuroticism"]].head())


# In[22]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["big5_neuroticism"], bins=30)
plt.title("Distribution of Big Five - Neuroticism")
plt.xlabel("Score of Neuroticism")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# #### Big Five

# In[23]:


# Big Five

big5_features = [
    "big5_openness",
    "big5_conscientiousness",
    "big5_extraversion",
    "big5_agreeableness",
    "big5_neuroticism"
]


# In[24]:


print("Big Five:")
display(user_features[big5_features].head())


# #### Big Five Correlation Matrix

# In[25]:


# Big Five Correlation Matrix

big5_cols = [
    "big5_openness",
    "big5_conscientiousness",
    "big5_extraversion",
    "big5_agreeableness",
    "big5_neuroticism",
]
corr = user_features[big5_cols].corr()


# In[26]:


print("Big Five Correlation Matrix:")
display(corr)


# In[27]:


plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(big5_cols)), big5_cols, rotation=45)
plt.yticks(range(len(big5_cols)), big5_cols)
plt.colorbar(label="Correlation")
plt.title("Big Five Correlation Matrix")
plt.tight_layout()
plt.show()


# ### Feature Creation: Gender

# In[28]:


category_features = [
    "restaurant_spend_share",
    "grocery_spend_share",
    "pharmacy_spend_share",
    "general_retail_spend_share",
    "family_spend_share",
    "fuel_spend_share"
]

big5_features = [
    "big5_openness",
    "big5_conscientiousness",
    "big5_extraversion",
    "big5_agreeableness",
    "big5_neuroticism"
]

# Combine all features we want to use for gender scoring
gender_signal_features = category_features + big5_features


# In[29]:


user_features["gender_raw_score"] = (
    + user_features["grocery_spend_share"]        # Women tend to score higher
    + user_features["pharmacy_spend_share"]
    + user_features["family_spend_share"]
    + user_features["general_retail_spend_share"]
    + user_features["big5_agreeableness"] * 0.4   # Women higher on average
    + user_features["big5_neuroticism"] * 0.4     # Women higher on average
    - user_features["fuel_spend_share"] * 0.8     # Men tend to score higher
    - user_features["restaurant_spend_share"] * 0.2  # Sometimes slightly male-skewed online
)


# In[30]:


user_features["gender_prob_female"] = 1 / (1 + np.exp(-user_features["gender_raw_score"]))
user_features["gender_prob_male"] = 1 - user_features["gender_prob_female"]


# In[31]:


user_features["gender_label"] = np.where(
    user_features["gender_prob_female"] >= 0.5,
    "female",
    "male"
)


# In[32]:


# Gender

gender_features = [
    "gender_prob_female",
    "gender_prob_male",
    "gender_label"
]


# In[33]:


print("Gender:")
display(user_features[gender_features].head())


# In[34]:


display(user_features["gender_label"].value_counts())


# ### Feature Creation: Age

# In[35]:


# Raw proxies
family_minus_social = user_features["family_spend_share"] - user_features["restaurant_spend_share"]
age_p1 = min_max_scale(family_minus_social)

log_total_spend = np.log1p(user_features["total_spend"])
age_p2 = min_max_scale(log_total_spend)

age_p3 = min_max_scale(user_features["loyalty_txn_fraction"])


# In[36]:


# Combine into age_score
user_features["age_score"] = (age_p1 + age_p2 + age_p3) / 3.0


# In[37]:


# Age groups via quantiles (relative to this population)
q = user_features["age_score"].quantile([0.25, 0.5, 0.75])
q25, q50, q75 = q[0.25], q[0.5], q[0.75]

def assign_age_group(score):
    if score <= q25:
        return "young_adult"
    elif score <= q50:
        return "early_middle"
    elif score <= q75:
        return "middle"
    else:
        return "older"

user_features["age_group"] = user_features["age_score"].apply(assign_age_group)


# In[38]:


print("Age:")
display(user_features[["age_score", "age_group"]].head())


# In[39]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["age_score"], bins=30)
plt.title("Distribution of Age Score")
plt.xlabel("Age Score (0=Young, 1=Old)")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# In[40]:


order = ["young_adult", "early_middle", "middle", "older"]

plt.figure(figsize=(10, 5))
user_features["age_group"].value_counts().reindex(order).plot(kind="bar")
plt.title("Distribution of Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# ### Feature Creation: Income Level

# In[41]:


income_p1 = min_max_scale(np.log1p(user_features["total_spend"]))
income_p2 = min_max_scale(user_features["avg_transaction_amount"])
income_p3 = min_max_scale(user_features["credit_card_txn_fraction"])


# In[42]:


if "city_Jakarta_share" in user_features.columns:
    income_p4 = min_max_scale(user_features["city_Jakarta_share"])
else:
    income_p4 = pd.Series(0.5, index=user_features.index)

user_features["income_score"] = (income_p1 + income_p2 + income_p3 + income_p4) / 4.0


# In[43]:


# Income bands via quantiles
iq = user_features["income_score"].quantile([0.25, 0.5, 0.75])
i25, i50, i75 = iq[0.25], iq[0.5], iq[0.75]

def assign_income_level(score):
    if score <= i25:
        return "low"
    elif score <= i50:
        return "lower_mid"
    elif score <= i75:
        return "upper_mid"
    else:
        return "high"

user_features["income_level"] = user_features["income_score"].apply(assign_income_level)


# In[44]:


print("Income:")
display(user_features[["income_score", "income_level"]].head())


# In[45]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["income_score"], bins=30)
plt.title("Distribution of Income Score")
plt.xlabel("Income Score")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# In[46]:


order = ["low", "lower_mid", "upper_mid", "high"]

plt.figure(figsize=(10, 5))
user_features["income_level"].value_counts().reindex(order).plot(kind="bar")
plt.title("Distribution of Income Level")
plt.xlabel("Income Level")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# ### Feature Creation: Educational Background

# In[47]:


edu_p1 = min_max_scale(user_features["category_entropy"])
edu_p2 = min_max_scale(user_features["merchant_entropy"])
edu_p3 = min_max_scale(user_features["income_score"])
edu_p4 = min_max_scale(user_features["loyalty_txn_fraction"])

user_features["education_score"] = (edu_p1 + edu_p2 + edu_p3 + edu_p4) / 4.0

eq = user_features["education_score"].quantile([0.33, 0.66])
e33, e66 = eq[0.33], eq[0.66]


# In[48]:


def assign_edu_level(score):
    if score <= e33:
        return "low"
    elif score <= e66:
        return "medium"
    else:
        return "high"

user_features["education_level"] = user_features["education_score"].apply(assign_edu_level)


# In[49]:


print("Education:")
display(user_features[["education_score", "education_level"]].head())


# In[50]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["education_score"], bins=30)
plt.title("Distribution of Education Score")
plt.xlabel("Education Score")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# In[51]:


order = ["low", "medium", "high"]

plt.figure(figsize=(10, 5))
user_features["education_level"].value_counts().reindex(order).plot(kind="bar")
plt.title("Distribution of Education Level")
plt.xlabel("Education Level")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# ### Feature Creation: Home Location

# In[52]:


user_features["home_city"] = user_features["primary_city"]
user_features["home_lat"] = user_features["centroid_lat"]
user_features["home_lon"] = user_features["centroid_lon"]


# In[ ]:


# display(user_features["home_city"].value_counts())


# In[53]:


print("Home Location:")
display(user_features[["home_city", "home_lat", "home_lon"]].head())


# ### Feature Creation: Work Location

# In[54]:


# Make sure all city_share columns are numeric
city_share_cols = [c for c in user_features.columns
                if c.startswith("city_") and c.endswith("_share")]

user_features[city_share_cols] = user_features[city_share_cols].apply(
    pd.to_numeric, errors="coerce"
)


# In[55]:


def infer_work_city(row):
    if row["n_cities"] <= 1:
        return row["home_city"]

    # Coerce row slice to numeric to avoid dtype object issues
    shares = pd.to_numeric(row[city_share_cols], errors="coerce")

    # If all shares are NaN or there is only one, fall back
    if shares.notna().sum() < 2:
        return row["home_city"]

    top2 = shares.nlargest(2)
    second_city_col = top2.index[1]

    return second_city_col.replace("city_", "").replace("_share", "")

def infer_work_reliability(row):
    return row["n_cities"] > 1


# In[56]:


user_features["work_city"] = user_features.apply(infer_work_city, axis=1)
user_features["work_location_reliable"] = user_features.apply(
    infer_work_reliability, axis=1
)


# In[ ]:


# display(user_features["work_city"].value_counts())


# In[57]:


print("Work Location:")
display(user_features[["home_city", "work_city", "n_cities", "work_location_reliable"]].head())


# ### 2nd Priority

# ### Feature Creation: Working Status

# In[58]:


def infer_working_status(row):
    weekend = row["weekend_txn_fraction"]
    income = row["income_score"]
    n_txn = row["n_transactions"]
    recency = row["recency_days"]
    
    # Very low engagement and long recency
    if n_txn < 5 and recency > 40:
        return "inactive_or_unemployed"
    
    # Higher income, moderate-to-high activity, more weekday than weekend
    if income > 0.6 and n_txn >= 10 and weekend < 0.4:
        return "full_time_or_professional"
    
    # Lower income, weekend-heavy behaviour
    if income < 0.4 and weekend > 0.5:
        return "student_or_part_time"
    
    return "flexible_or_unknown"


# In[59]:


user_features["working_status"] = user_features.apply(infer_working_status, axis=1)


# In[ ]:


# display(user_features["working_status"].value_counts())


# In[60]:


print("Working Status:")
display(user_features[["working_status", "income_score", "weekend_txn_fraction", "n_transactions", "recency_days"]].head())


# In[76]:


plt.figure(figsize=(10, 5))
user_features["working_status"].value_counts().plot(kind="bar")
plt.title("Distribution of Working Status")
plt.xlabel("Working Status")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# ### Feature Creation: Industry of Employment

# In[65]:


# Pick dominant MCC by spend among top 10 MCCs we already have columns for
mcc_spend_cols = [c for c in user_features.columns if c.startswith("cat_") and c.endswith("_spend_share")]


# In[68]:


def infer_dominant_mcc(row):
    share_cols = [c for c in row.index
                if c.startswith("cat_") and c.endswith("_spend_share")]
    shares = row[share_cols]

    # Force numeric; invalid entries become NaN
    shares = pd.to_numeric(shares, errors="coerce")

    if shares.isna().all():
        return np.nan
    
    col = shares.idxmax()
    # "cat_XXXX_spend_share" -> XXXX
    return int(col.replace("cat_", "").replace("_spend_share", ""))


# In[69]:


user_features["dominant_mcc"] = user_features.apply(infer_dominant_mcc, axis=1)


# In[70]:


# Map MCC to coarse industry
mcc_to_industry = {
    5411: "retail_grocery",
    5412: "retail_grocery",
    5812: "hospitality_food",
    5814: "hospitality_food",
    5912: "health_pharmacy",
    5732: "electronics_media",
    5999: "general_retail",
    5942: "books_stationery",
    5541: "transport_auto",
    5251: "home_improvement",
}

def map_industry(mcc):
    if np.isnan(mcc):
        return "unknown"
    return mcc_to_industry.get(int(mcc), "other")


# In[71]:


user_features["industry_affinity"] = user_features["dominant_mcc"].apply(map_industry)


# In[74]:


# display(user_features["industry_affinity"].value_counts())


# In[72]:


print("Industry of Employment:")
display(user_features[["dominant_mcc", "industry_affinity"]].head())


# In[75]:


plt.figure(figsize=(10, 5))
user_features["industry_affinity"].value_counts().plot(kind="bar")
plt.title("Distribution of Industry of Employment")
plt.xlabel("Industry of Employment")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# ### Feature Creation: Health Status

# In[ ]:


health_positive = user_features["pharmacy_spend_share"] + 0.5 * user_features["grocery_spend_share"]
health_negative = user_features["restaurant_spend_share"]


# In[78]:


health_raw = health_positive - health_negative
user_features["health_behavior_score"] = min_max_scale(health_raw)


# In[79]:


def assign_health_label(score):
    if score >= 0.66:
        return "health_oriented"
    elif score <= 0.33:
        return "less_health_oriented"
    else:
        return "neutral"


# In[80]:


user_features["health_status"] = user_features["health_behavior_score"].apply(assign_health_label)


# In[81]:


print("Health Status:")
display(user_features[["health_behavior_score", "health_status"]].head())


# In[82]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["health_behavior_score"], bins=30)
plt.title("Distribution of Health Behaviour Score")
plt.xlabel("Health Behaviour Score")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# In[83]:


plt.figure(figsize=(10, 5))
user_features["health_status"].value_counts().plot(kind="bar")
plt.title("Distribution of Health Status")
plt.xlabel("Health Status")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# ### Feature Creation: Homeownership Status

# In[84]:


user_features["homeownership_status"] = "unknown"


# In[85]:


print("Homeownership Status:")
display(user_features[["homeownership_status"]].head())


# ### Feature Creation: Marital Status

# In[86]:


marital_raw = family_minus_social + user_features["age_score"]
user_features["marital_score"] = min_max_scale(marital_raw)


# In[87]:


def assign_marital_status(score):
    if score >= 0.7:
        return "married_or_partnered_likely"
    elif score <= 0.3:
        return "single_likely"
    else:
        return "unknown"


# In[88]:


user_features["marital_status"] = user_features["marital_score"].apply(assign_marital_status)


# In[89]:


print("Marital Status:")
display(user_features[["marital_score", "marital_status"]].head())


# In[90]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["marital_score"], bins=30)
plt.title("Distribution of Marital Score")
plt.xlabel("Marital Score")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# In[91]:


plt.figure(figsize=(10, 5))
user_features["marital_status"].value_counts().plot(kind="bar")
plt.title("Distribution of Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# ### Feature Creation: Parental Status

# In[92]:


parent_raw = user_features["family_spend_share"] * user_features["age_score"]
user_features["parental_score"] = min_max_scale(parent_raw)


# In[93]:


def assign_parental_status(score):
    if score >= 0.7:
        return "parent_likely"
    elif score <= 0.3:
        return "non_parent_likely"
    else:
        return "unknown"


# In[94]:


user_features["parental_status"] = user_features["parental_score"].apply(assign_parental_status)


# In[95]:


print("Parental Status:")
display(user_features[["parental_score", "parental_status"]].head())


# In[96]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["parental_score"], bins=30)
plt.title("Distribution of Parental Score")
plt.xlabel("Parental Score")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# In[97]:


plt.figure(figsize=(10, 5))
user_features["parental_status"].value_counts().plot(kind="bar")
plt.title("Distribution of Parental Status")
plt.xlabel("Parental Status")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# ### Feature Creation: Vehicle Ownership

# In[98]:


veh_signal = user_features["fuel_spend_share"]
user_features["vehicle_ownership_score"] = min_max_scale(veh_signal)


# In[99]:


def assign_vehicle_status(score):
    if score >= 0.6:
        return "owner_likely"
    elif score <= 0.1:
        return "non_owner_likely"
    else:
        return "unknown"


# In[100]:


user_features["vehicle_ownership"] = user_features["vehicle_ownership_score"].apply(assign_vehicle_status)


# In[101]:


print("Vehicle Ownership:")
display(user_features[["vehicle_ownership_score", "vehicle_ownership"]].head())


# In[102]:


plt.figure(figsize=(10, 5))
plt.hist(user_features["vehicle_ownership_score"], bins=30)
plt.title("Distribution of Vehicle Ownership Score")
plt.xlabel("Vehicle Ownership Score")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# In[103]:


plt.figure(figsize=(10, 5))
user_features["vehicle_ownership"].value_counts().plot(kind="bar")
plt.title("Distribution of Vehicle Ownership")
plt.xlabel("Vehicle Ownership")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()


# ### Data Checking

# In[104]:


print("User-Level Features:", user_features.shape)
print(user_features.head())
# display(user_features.head())


# In[105]:


print(user_features.isnull().sum())
# display(user_features.isnull().sum())


# In[106]:


display(user_features.info())


# ### Data Exporting

# In[107]:


# user_features.to_csv('./data/transactions_eda_ulf.csv', index=False)
user_features.to_parquet("./data/transactions_eda_df.parquet", index=False)


# In[109]:


get_ipython().run_line_magic('reset', '-f')


# ## Feature Engineering: Total Features (for Segmentation)

# In[1]:


get_ipython().run_line_magic('reset', '-f')


# In[20]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)


# ### Data Importing

# In[3]:


# user_features = pd.read_csv('./data/transactions_eda_df.csv')
user_features = pd.read_parquet('./data/transactions_eda_df.parquet')


# In[4]:


display(user_features.info())


# ### Data Pre-Processing

# In[5]:


user_features = user_features.copy()
user_features["user_index"] = np.arange(len(user_features))


# In[6]:


print("user_features with user_index:", user_features.shape)
display(user_features[["user_index"]].head())


# ### Data Segmentation

# In[8]:


# Numeric features focused on behaviour + profile scores
seg_numeric_features = [
    # Spend / Volume + Completion / Refund + Temporal
    "n_transactions",
    "total_spend",
    "avg_transaction_amount",

    # Temporal
    "activity_span_days",
    "n_active_days",
    "avg_txn_per_active_day",
    "recency_days",
    "weekend_txn_fraction",
    "weekday_txn_fraction",

    # Merchant
    "restaurant_spend_share",
    "grocery_spend_share",
    "pharmacy_spend_share",
    "general_retail_spend_share",
    "fuel_spend_share",
    "family_spend_share",

    "n_distinct_categories",
    "n_distinct_merchants",
    "category_entropy",
    "merchant_entropy",
    "top_merchant_spend_share",

    # Location
    "n_distinct_locations",
    "avg_distance_from_centroid_km",

    # Payment
    "credit_card_txn_fraction",
    "balance_txn_fraction",
    "promo_to_spend_ratio",
    "loyalty_txn_fraction",

    # Ratings
    "avg_rating",
    "rating_std",
    "n_low_ratings",
    "n_high_ratings",

    # Big Five Personality Score
    "big5_openness",
    "big5_conscientiousness",
    "big5_extraversion",
    "big5_agreeableness",
    "big5_neuroticism",

    # Age
    "age_score",
    # Income Level
    "income_score",
    # Educational Background
    "education_score",
    # Health Status
    "health_behavior_score",
    # Vehicle Ownership
    "vehicle_ownership_score",
]


# In[9]:


# Categorical features
seg_categorical_features = [
    # Home Location
    "home_city",
    # Work Location
    "work_city",
    # Working Status
    "working_status",
    # Industry of Employment
    "industry_affinity",
    # Income Level
    "income_level",
    # Age
    "age_group",
    # Educational Background
    "education_level",
    # Health Status
    "health_status",
    # Vehicle Ownership
    "vehicle_ownership",
    # Device
    "primary_device_family",
]


# In[10]:


print("Number of Numeric Features:", len(seg_numeric_features))
print("Number of Categorical Features:", len(seg_categorical_features))


# In[ ]:


# Build base segmentation DataFrame
seg_cols = seg_numeric_features + seg_categorical_features
segmentation_df = user_features[seg_cols].copy()


# In[13]:


print("Segmentation DataFrame:", segmentation_df.shape)
display(segmentation_df.head())


# In[14]:


# Replace inf with NaN on numeric columns, then fill with median
segmentation_df[seg_numeric_features] = (
    segmentation_df[seg_numeric_features]
    .replace([np.inf, -np.inf], np.nan)
)


# In[15]:


numeric_medians = segmentation_df[seg_numeric_features].median()
segmentation_df[seg_numeric_features] = segmentation_df[seg_numeric_features].fillna(numeric_medians)


# In[16]:


# Categorical: fill missing with 'unknown'
for col in seg_categorical_features:
    segmentation_df[col] = segmentation_df[col].fillna("unknown").astype(str)


# ### Data Visualization

# In[18]:


key_hist_features = [
    "n_transactions", "total_spend", "avg_transaction_amount",
    "weekend_txn_fraction", "restaurant_spend_share",
    "grocery_spend_share", "loyalty_txn_fraction",
    "income_score", "age_score", "education_score",
]


# In[ ]:


for col in key_hist_features:
    if col not in segmentation_df.columns:
        continue
    plt.figure(figsize=(10, 5))
    plt.hist(segmentation_df[col], bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.show()


# ### Data Scaling & Data Encoding

# In[21]:


# Standardize numeric features (mean 0, std 1)
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(segmentation_df[seg_numeric_features])


# In[24]:


print("Shape of Scaled Numeric Features:", X_numeric_scaled.shape)


# In[23]:


# One-hot encode categorical features
X_categorical = pd.get_dummies(
    segmentation_df[seg_categorical_features],
    drop_first=False
)


# In[25]:


print("Shape of Scaled Categorical Features", X_categorical.shape)


# In[26]:


# Combine numeric + categorical into one matrix
X_combined = np.hstack([X_numeric_scaled, X_categorical.values])
seg_feature_names = seg_numeric_features + list(X_categorical.columns)

X = pd.DataFrame(
    X_combined,
    columns=seg_feature_names,
    index=user_features.index
)


# In[27]:


print("Shape of Combined Features:", X.shape)
display(X.head())


# ### Data Exporting

# In[28]:


X.to_parquet("./data/transactions_eda_tf.parquet", index=False)


# In[1]:


get_ipython().run_line_magic('reset', '-f')


# ## Feature Segmentation

# In[2]:


get_ipython().run_line_magic('reset', '-f')


# In[3]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)


# ### Data Importing

# In[4]:


user_features = pd.read_parquet('./data/transactions_eda_df.parquet')


# In[5]:


display(user_features.info())


# In[6]:


X = pd.read_parquet('./data/transactions_eda_tf.parquet')


# In[7]:


display(X.info())


# ### Data Clustering w/ K-Means

# In[8]:


k_values = [4, 5, 6, 7, 8, 9]
inertias = []
silhouette_scores = []


# In[9]:


# Use a sample for silhouette to keep computation manageable
n_samples_for_sil = min(5000, X.shape[0])
rng = np.random.RandomState(42)
sample_indices = rng.choice(X.shape[0], size=n_samples_for_sil, replace=False)
X_sample = X.iloc[sample_indices]


# In[10]:


for k in k_values:
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    km.fit(X)

    inertias.append(km.inertia_)

    # Silhouette on the sample
    labels_sample = km.labels_[sample_indices]
    sil = silhouette_score(X_sample, labels_sample)
    silhouette_scores.append(sil)
    print(f"k={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")


# In[13]:


plt.figure(figsize=(10, 5))
plt.plot(k_values, inertias, marker="o")
plt.title("K-Means Elbow Plot")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster SSE)")
plt.xticks(k_values)
plt.tight_layout()
plt.show()


# In[15]:


plt.figure(figsize=(10, 5))
plt.plot(k_values, silhouette_scores, marker="o")
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.xticks(k_values)
plt.tight_layout()
plt.show()


# In[16]:


k_opt = 4
# It gives the highest silhouette score, indicating the most meaningful and well-separated clusters.
# Adding more clusters reduces silhouette quality without giving a clear elbow improvement.

kmeans_final = KMeans(
    n_clusters=k_opt,
    random_state=42,
    n_init=20,
    max_iter=500
)
kmeans_final.fit(X)


# In[18]:


cluster_labels = kmeans_final.labels_

unique, counts = np.unique(cluster_labels, return_counts=True)
for c, cnt in zip(unique, counts):
    print(f"Cluster {c}: {cnt} Users")


# In[ ]:


user_features = user_features
user_features["cluster_kmeans"] = cluster_labels


# In[20]:


print("user_features w/ Clusters:", user_features.shape)
display(user_features[["cluster_kmeans"]].head())


# In[27]:


cluster_profile_features = [
    # Spend / Volume + Completion / Refund + Temporal
    "n_transactions",
    "total_spend",
    "avg_transaction_amount",

    # Temporal
    "activity_span_days",
    "n_active_days",
    "avg_txn_per_active_day",
    "recency_days",
    "weekend_txn_fraction",
    "weekday_txn_fraction",

    # Merchant
    "restaurant_spend_share",
    "grocery_spend_share",
    "pharmacy_spend_share",
    "general_retail_spend_share",
    "fuel_spend_share",
    "family_spend_share",

    "n_distinct_categories",
    "n_distinct_merchants",
    "category_entropy",
    "merchant_entropy",
    "top_merchant_spend_share",

    # Location
    "n_distinct_locations",
    "avg_distance_from_centroid_km",

    # Payment
    "credit_card_txn_fraction",
    "balance_txn_fraction",
    "promo_to_spend_ratio",
    "loyalty_txn_fraction",

    # Big Five Personality Score
    "big5_openness",
    "big5_conscientiousness",
    "big5_extraversion",
    "big5_agreeableness",
    "big5_neuroticism",

    # Age
    "age_score",
    # Income Level
    "income_score",
    # Educational Background
    "education_score",
    # Health Status
    "health_behavior_score",
    # Vehicle Ownership
    "vehicle_ownership_score",
]


# In[28]:


cluster_summary = (
    user_features
    .groupby("cluster_kmeans")[cluster_profile_features]
    .mean()
    .round(3)
)


# In[42]:


print("Cluster Summary:", cluster_summary.shape)
print(cluster_summary)


# In[33]:


cat_profile_cols = [
    "income_level",
    "age_group",
    "working_status",
    "education_level",
    "health_status",
    "vehicle_ownership",
    "industry_affinity",
    "home_city",
    "work_city",
]


# In[43]:


for col in cat_profile_cols:
    if col not in user_features.columns:
        continue
    print(f"\nCluster vs {col}")
    ct = pd.crosstab(
        user_features["cluster_kmeans"],
        user_features[col],
        normalize="index"
    ).round(3)
    print(ct)


# ### Data Visualization w/ PCA

# In[38]:


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)


# In[39]:


print("PCA Explained Variance Ratios:", pca.explained_variance_ratio_)


# In[44]:


pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=user_features.index
)
pca_df["cluster_kmeans"] = cluster_labels

cluster_pca_means = (
    pca_df.groupby("cluster_kmeans")[["PC1", "PC2"]].mean()
)


# In[45]:


print("Cluster Centroids in PCA:")
display(cluster_pca_means.round(3))


# In[50]:


plt.figure(figsize=(10, 10))
plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=cluster_labels,
    s=5,
    alpha=0.3,
    cmap="tab10"
)

for cluster_id, row in cluster_pca_means.iterrows():
    plt.scatter(row["PC1"], row["PC2"], c="black", s=80, marker="X")
    plt.text(row["PC1"] + 0.1, row["PC2"] + 0.1, f"C{cluster_id}", fontsize=9)

plt.title("User Segmentation (color: cluster)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()


# ### Data Exporting

# In[51]:


user_features.to_parquet("./data/transactions_eda_s.parquet", index=False)


# In[52]:


get_ipython().run_line_magic('reset', '-f')

