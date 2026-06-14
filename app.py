"""
🛒 Shopper Spectrum – Streamlit App
Modules:
  1. Product Recommendation (Collaborative Filtering)
  2. Customer Segmentation (RFM + KMeans)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛒 Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #555; font-size: 1rem; margin-bottom: 2rem; }
    .segment-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 16px; padding: 20px 28px;
        text-align: center; margin-top: 1.2rem;
    }
    .segment-card h2 { font-size: 2rem; margin: 0; }
    .segment-card p  { font-size: 0.95rem; margin: 4px 0 0 0; opacity: 0.85; }
    .rec-card {
        background: #f8f9fe; border-left: 5px solid #667eea;
        border-radius: 8px; padding: 10px 16px; margin: 6px 0;
    }
    .rec-rank { color: #667eea; font-weight: 700; }
    .metric-box {
        background: #f0f2ff; border-radius: 12px;
        padding: 14px 18px; margin: 6px 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.4rem; font-weight: 600; font-size: 1rem;
    }
    .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = "models"
    with open(f"{base}/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(f"{base}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{base}/label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    similarity_df = pd.read_pickle(f"{base}/similarity_df.pkl")
    rfm = pd.read_csv(f"{base}/rfm_segments.csv")
    return kmeans, scaler, label_map, similarity_df, rfm


kmeans, scaler, label_map, similarity_df, rfm = load_models()

product_list = sorted(similarity_df.index.tolist())

# ── Segment metadata ──────────────────────────────────────────────────────────
SEGMENT_META = {
    "High-Value": {
        "emoji": "💎",
        "color": "#E91E63",
        "description": "Recent, frequent, and high-spending customers. Reward them with VIP perks.",
        "action": "🎁 Offer exclusive loyalty rewards & early access to new products.",
    },
    "Regular": {
        "emoji": "⭐",
        "color": "#2196F3",
        "description": "Steady purchasers with moderate spend. Good engagement potential.",
        "action": "📧 Send personalised upsell campaigns and bundle offers.",
    },
    "Occasional": {
        "emoji": "🛍️",
        "color": "#4CAF50",
        "description": "Infrequent buyers with lower spend. Needs nurturing.",
        "action": "💌 Trigger re-engagement emails with seasonal discounts.",
    },
    "At-Risk": {
        "emoji": "⚠️",
        "color": "#FF9800",
        "description": "Haven't purchased in a long time. Risk of churn is high.",
        "action": "🆘 Win-back campaign: time-limited discount + personal outreach.",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_recommendations(product_name: str, n: int = 5):
    pname = product_name.strip().upper()
    if pname not in similarity_df.index:
        matches = [p for p in similarity_df.index if pname in p]
        if not matches:
            return None, pname
        pname = matches[0]
    recs = (
        similarity_df[pname]
        .drop(pname)
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    recs.columns = ["Product", "Similarity"]
    recs["Similarity"] = recs["Similarity"].round(4)
    return recs, pname


def predict_segment(recency: float, frequency: float, monetary: float):
    x = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans.predict(x)[0]
    segment = label_map[cluster]
    return segment, cluster


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛒 Shopper Spectrum")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🎯 Product Recommendations", "👥 Customer Segmentation"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Dataset Stats**")
st.sidebar.metric("Customers Segmented", f"{len(rfm):,}")
st.sidebar.metric("Products in Catalogue", f"{len(similarity_df):,}")
st.sidebar.metric("Clusters (k)", "4")


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-header">🛒 Shopper Spectrum</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Customer Segmentation & Product Recommendations for E-Commerce</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Product Recommendations")
        st.write(
            "Enter a product name and get **5 similar products** "
            "using item-based collaborative filtering with cosine similarity."
        )
        if st.button("Go to Recommendations ➜"):
            st.session_state["page"] = "🎯 Product Recommendations"

    with col2:
        st.markdown("### 👥 Customer Segmentation")
        st.write(
            "Enter a customer's **Recency, Frequency, and Monetary** values "
            "to instantly predict their segment: High-Value, Regular, Occasional, or At-Risk."
        )
        if st.button("Go to Segmentation ➜"):
            st.session_state["page"] = "👥 Customer Segmentation"

    st.markdown("---")
    st.markdown("### 🏷️ Customer Segments at a Glance")
    cols = st.columns(4)
    for col, (seg, meta) in zip(cols, SEGMENT_META.items()):
        with col:
            seg_count = (rfm["Segment"] == seg).sum()
            pct = seg_count / len(rfm) * 100
            st.markdown(
                f"""<div class='metric-box'>
                <b style='color:{meta["color"]}'>{meta["emoji"]} {seg}</b><br>
                <span style='font-size:1.5rem;font-weight:700'>{seg_count:,}</span>
                <span style='font-size:0.85rem;color:#666'> customers ({pct:.1f}%)</span><br>
                <small>{meta["description"]}</small>
                </div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Product Recommendations":
    st.markdown("## 🎯 Product Recommendation Engine")
    st.write(
        "Type a product name (or select from the dropdown) and click **Get Recommendations** "
        "to see the 5 most similar products based on customer purchase patterns."
    )
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        product_input = st.selectbox(
            "🔍 Select or type a product name:",
            options=[""] + product_list,
            index=0,
            help="Products are from the top-150 best-sellers in the dataset.",
        )
    with col2:
        n_recs = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    run = st.button("🚀 Get Recommendations")

    if run:
        if not product_input:
            st.warning("⚠️ Please select or type a product name.")
        else:
            recs, matched = get_recommendations(product_input, n=n_recs)
            if recs is None:
                st.error(f"❌ Product **{product_input}** not found. Try a different name.")
            else:
                if matched != product_input.strip().upper():
                    st.info(f"ℹ️ Matched to closest product: **{matched}**")
                st.success(f"✅ Top {n_recs} products similar to: **{matched}**")
                st.markdown("---")
                for i, row in recs.iterrows():
                    bar_width = int(row["Similarity"] * 100)
                    st.markdown(
                        f"""<div class='rec-card'>
                        <span class='rec-rank'>#{i+1}</span>&nbsp;&nbsp;
                        <b>{row["Product"]}</b><br>
                        <small>Similarity score: <b>{row["Similarity"]:.4f}</b></small>
                        <div style='background:#e0e0e0;border-radius:4px;height:6px;margin-top:6px'>
                          <div style='background:#667eea;width:{bar_width}%;height:6px;border-radius:4px'></div>
                        </div></div>""",
                        unsafe_allow_html=True,
                    )
                st.markdown("---")
                st.dataframe(recs.rename(columns={"Product": "Recommended Product", "Similarity": "Cosine Similarity"}), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.markdown("## 👥 Customer Segmentation Predictor")
    st.write(
        "Enter a customer's **RFM** values to predict which segment they belong to "
        "using the trained KMeans model."
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input(
            "📅 Recency (days since last purchase)",
            min_value=0, max_value=730, value=30, step=1,
            help="Lower = more recent purchase. 0 = bought today."
        )
    with col2:
        frequency = st.number_input(
            "🔁 Frequency (number of orders)",
            min_value=1, max_value=500, value=10, step=1,
            help="Total number of distinct invoices for this customer."
        )
    with col3:
        monetary = st.number_input(
            "💰 Monetary (total spend in £)",
            min_value=1.0, max_value=100000.0, value=500.0, step=10.0,
            help="Total amount the customer has spent."
        )

    predict_btn = st.button("🔮 Predict Cluster")

    if predict_btn:
        segment, cluster = predict_segment(recency, frequency, monetary)
        meta = SEGMENT_META[segment]

        st.markdown(
            f"""<div class='segment-card' style='background:linear-gradient(135deg,{meta["color"]}cc,{meta["color"]}88)'>
            <h2>{meta["emoji"]} {segment} Customer</h2>
            <p>{meta["description"]}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recency", f"{recency} days")
        c2.metric("Frequency", f"{frequency} orders")
        c3.metric("Monetary", f"£{monetary:,.2f}")
        c4.metric("Cluster", f"#{cluster}")

        st.markdown(f"### 💡 Recommended Action")
        st.info(meta["action"])

        # Show where this customer sits vs the distribution
        st.markdown("### 📊 How This Customer Compares")
        col_r, col_f, col_m = st.columns(3)

        for col, field, val, label in [
            (col_r, "Recency", recency, "Recency (days)"),
            (col_f, "Frequency", frequency, "Frequency"),
            (col_m, "Monetary", monetary, "Monetary (£)"),
        ]:
            pct = int((rfm[field] <= val).mean() * 100)
            with col:
                st.metric(label, f"{val}", delta=f"Better than {pct}% of customers" if field != "Recency" else f"More recent than {100-pct}%")
