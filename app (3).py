import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from scipy import sparse

# --- Configuration (matching notebook settings) ---
# These paths should point to where your cached files are located.
# Update RUN_SUFFIX if you used a different sampling strategy.
RUN_SUFFIX = "_sample100000" # Or "_full" if you ran the full pipeline

CLEANED_PATH = f"cleaned_ratings_base{RUN_SUFFIX}.parquet"
TRAIN_PATH = f"interactions_train{RUN_SUFFIX}.parquet"
FEATURES_PATH = f"user_features_final{RUN_SUFFIX}.parquet"
ASSIGN_PATH = f"user_cluster_assignments{RUN_SUFFIX}.parquet"
MAPPINGS_PATH = f"id_mappings{RUN_SUFFIX}.pkl"
SIM_IDX_PATH = f"item_top_n_indices{RUN_SUFFIX}.npy"
SIM_SCORE_PATH = f"item_top_n_scores{RUN_SUFFIX}.npy"
SPARSE_PATH = f"user_item_train_sparse{RUN_SUFFIX}.npz"

# Best CF weight determined from optimization (from d06a98d1 in notebook)
BEST_CF_WEIGHT = 0.5
TOP_K_RECS = 5 # Number of recommendations to display


@st.cache_data # Cache data loading for performance
def load_data():
    """Loads all necessary data and model artifacts."""
    print("Loading data and models...")
    cleaned_df = pd.read_parquet(CLEANED_PATH)
    train_df = pd.read_parquet(TRAIN_PATH)
    user_features = pd.read_parquet(FEATURES_PATH)
    user_cluster_assignments = pd.read_parquet(ASSIGN_PATH)

    with open(MAPPINGS_PATH, "rb") as f:
        mappings = pickle.load(f)
    user_to_idx = mappings["user_to_idx"]
    product_to_idx = mappings["product_to_idx"]
    idx_to_product = mappings["idx_to_product"]

    top_n_indices = np.load(SIM_IDX_PATH)
    top_n_scores = np.load(SIM_SCORE_PATH)
    user_item_train = sparse.load_npz(SPARSE_PATH).tocsr()

    # Recompute item_user_norm for diversity calculation (as in notebook)
    item_user = user_item_train.T.tocsr()
    norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    item_user_norm = (sparse.diags(1.0 / norms) @ item_user).tocsr()

    # Pre-calculate cluster popularity (as in notebook)
    train_with_cluster = train_df.merge(user_cluster_assignments, on="userId", how="left")
    cluster_popularity = {
        c: grp["productId"].value_counts().index.to_numpy()
        for c, grp in train_with_cluster.groupby("cluster", observed=True)
    }
    user_to_cluster = dict(zip(user_cluster_assignments["userId"], user_cluster_assignments["cluster"]))

    # Pre-calculate overall product popularity
    product_pop_train = train_df.groupby("productId", observed=True).size()

    print("Data and models loaded.")
    return (
        cleaned_df, train_df, user_features, user_cluster_assignments,
        user_to_idx, product_to_idx, idx_to_product, top_n_indices,
        top_n_scores, user_item_train, item_user_norm, cluster_popularity,
        user_to_cluster, product_pop_train
    )

# --- Recommendation Functions (adapted from notebook) ---
# These are placed here to be part of the Streamlit app's scope

def recommend_popularity(rated_product_idx_set, product_pop_train, product_to_idx, top_k=10):
    """Recommends top K most popular products not yet rated by the user."""
    popular_pids = product_pop_train.index.to_numpy()
    recs = []
    for pid in popular_pids:
        if pid in product_to_idx:
            if product_to_idx[pid] not in rated_product_idx_set:
                recs.append(pid)
                if len(recs) == top_k:
                    break
    return recs

def recommend_cluster(user_id, rated_product_idx_set, cluster_popularity, user_to_cluster, product_to_idx, product_pop_train, top_k=10):
    """Most popular products within the user's own behavioral cluster."""
    cluster = user_to_cluster.get(user_id)
    if cluster is None:
        return recommend_popularity(rated_product_idx_set, product_pop_train, product_to_idx, top_k)

    recs = []
    valid_pids = [pid for pid in cluster_popularity[cluster] if pid in product_to_idx]
    for pid in valid_pids:
        if product_to_idx[pid] not in rated_product_idx_set:
            recs.append(pid)
            if len(recs) == top_k:
                break
    return recs

def recommend_item_cf(user_idx, user_item_train, top_n_indices, top_n_scores, idx_to_product, top_k=10):
    """Recommends top K items based on item-item collaborative filtering."""
    user_row = user_item_train[user_idx]
    rated_items = set(user_row.indices)

    item_scores = {}
    for item_rated_idx, rating in zip(user_row.indices, user_row.data):
        if item_rated_idx < len(top_n_indices) and item_rated_idx < len(top_n_scores):
            for n_idx, sim_score in zip(top_n_indices[item_rated_idx], top_n_scores[item_rated_idx]):
                if n_idx not in rated_items:
                    item_scores[n_idx] = item_scores.get(n_idx, 0.0) + sim_score * rating

    ranked_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    recs = [idx_to_product[idx] for idx, _ in ranked_items[:top_k] if idx in idx_to_product]
    return recs

def recommend_hybrid(user_id, user_idx, user_item_train, top_n_indices, top_n_scores, cluster_popularity, user_to_cluster, product_to_idx, idx_to_product, product_pop_train, cf_weight, top_k=10):
    """Blend item-item CF score with in-cluster popularity, both normalized to [0,1]."""
    user_row = user_item_train[user_idx]
    rated_set = set(user_row.indices)

    cf_scores = {}
    for item_idx, rating in zip(user_row.indices, user_row.data):
        if item_idx < len(top_n_indices) and item_idx < len(top_n_scores):
            for n_idx, sim in zip(top_n_indices[item_idx], top_n_scores[item_idx]):
                if n_idx in rated_set or sim <= 0:
                    continue
                cf_scores[n_idx] = cf_scores.get(n_idx, 0.0) + sim * rating

    cluster = user_to_cluster.get(user_id)
    cluster_scores = {}
    if cluster is not None and cluster in cluster_popularity:
        ranked_ids = cluster_popularity[cluster][:200]
        max_rank = len(ranked_ids)
        for rank, pid in enumerate(ranked_ids):
            if pid in product_to_idx:
                p_idx = product_to_idx[pid]
                if p_idx not in rated_set:
                    cluster_scores[p_idx] = (max_rank - rank) / max_rank

    if not cf_scores and not cluster_scores:
        return recommend_popularity(rated_set, product_pop_train, product_to_idx, top_k)

    def normalize(d):
        m = max(d.values()) if d else 0
        return {k: v / m for k, v in d.items()} if m > 0 else d

    cf_norm, cluster_norm = normalize(cf_scores), normalize(cluster_scores)
    all_items = set(cf_norm) | set(cluster_norm)
    blended = {i: cf_weight * cf_norm.get(i, 0) + (1 - cf_weight) * cluster_norm.get(i, 0) for i in all_items}
    ranked = sorted(blended.items(), key=lambda x: -x[1])[:top_k]
    recs = [idx_to_product[idx] for idx, _ in ranked if idx in idx_to_product]
    return recs


# --- Streamlit UI ---
st.title("Product Recommendation System")
st.write("Select a user ID to get personalized product recommendations.")

# Load all data
(cleaned_df, train_df, user_features, user_cluster_assignments,
 user_to_idx, product_to_idx, idx_to_product, top_n_indices,
 top_n_scores, user_item_train, item_user_norm, cluster_popularity,
 user_to_cluster, product_pop_train) = load_data()

# Get all unique user IDs from the training set for selection
all_user_ids = sorted(user_to_idx.keys())

# User selection dropdown
selected_user_id = st.selectbox("Select a User ID", all_user_ids)

if selected_user_id:
    user_idx = user_to_idx.get(selected_user_id)
    if user_idx is None:
        st.warning("Selected User ID not found in training data for recommendations.")
    else:
        st.subheader(f"Recommendations for User: {selected_user_id}")

        # Display user profile stats (similar to notebook example)
        st.write("**User Profile Summary:**")
        user_profile_df = user_features[user_features["userId"] == selected_user_id]
        if not user_profile_df.empty:
            user_profile = user_profile_df.iloc[0]
            st.write(f"- Number of ratings: {user_profile['n_ratings']:.0f}")
            st.write(f"- Average rating: {user_profile['avg_rating']:.2f}")
            st.write(f"- Cluster: {user_to_cluster.get(selected_user_id, 'N/A')}")
            # Add more stats if desired
        else:
            st.write("No detailed user profile available.")

        st.write("\n---")
        st.subheader(f"Top {TOP_K_RECS} Hybrid Recommendations:")

        # Generate hybrid recommendations
        hybrid_recs = recommend_hybrid(
            selected_user_id, user_idx, user_item_train, top_n_indices, top_n_scores,
            cluster_popularity, user_to_cluster, product_to_idx, idx_to_product,
            product_pop_train, BEST_CF_WEIGHT, TOP_K_RECS
        )

        if hybrid_recs:
            for i, prod_id in enumerate(hybrid_recs):
                st.write(f"{i+1}. {prod_id}")
        else:
            st.write("No recommendations found for this user.")

        st.write("\n---")
        st.subheader("Other Recommendation Types for Comparison:")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Item-Item CF Recommendations:**")
            item_cf_recs = recommend_item_cf(
                user_idx, user_item_train, top_n_indices, top_n_scores, idx_to_product, TOP_K_RECS
            )
            if item_cf_recs:
                for i, prod_id in enumerate(item_cf_recs):
                    st.write(f"{i+1}. {prod_id}")
            else:
                st.write("No CF recommendations found.")

        with col2:
            st.write("**Cluster-Based Recommendations:**")
            rated_set_for_cluster = set(user_item_train[user_idx].indices) if user_idx is not None else set()
            cluster_recs = recommend_cluster(
                selected_user_id, rated_set_for_cluster, cluster_popularity, user_to_cluster,
                product_to_idx, product_pop_train, TOP_K_RECS
            )
            if cluster_recs:
                for i, prod_id in enumerate(cluster_recs):
                    st.write(f"{i+1}. {prod_id}")
            else:
                st.write("No cluster recommendations found.")
