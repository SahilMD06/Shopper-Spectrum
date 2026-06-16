"""
🛒 Shopper Spectrum – Streamlit App
Fixes:
  1. Navigation buttons on Home page now work (use st.query_params)
  2. Product names shown correctly (not codes)
  3. Improved recommendation UI with % match bars, badges, and better layout
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛒 Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── General ── */
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"] { background: #1a1d27; }
  .block-container { padding-top: 2rem; }

  /* ── Sidebar nav label ── */
  .sidebar-title {
    font-size: 1.3rem; font-weight: 800; color: #a78bfa;
    margin-bottom: 0.2rem; padding: 0 0.5rem;
  }

  /* ── Hero header ── */
  .hero-title {
    font-size: 2.8rem; font-weight: 900; letter-spacing: -1px;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero-sub { color: #94a3b8; font-size: 1.05rem; margin-top: -0.5rem; margin-bottom: 1.5rem; }

  /* ── Home feature cards ── */
  .feature-card {
    background: #1e2130; border: 1px solid #2d3148; border-radius: 16px;
    padding: 24px 28px; height: 100%; transition: border-color 0.2s;
  }
  .feature-card:hover { border-color: #a78bfa; }
  .feature-card h3 { color: #e2e8f0; margin-bottom: 0.5rem; font-size: 1.2rem; }
  .feature-card p  { color: #94a3b8; font-size: 0.9rem; line-height: 1.6; }

  /* ── Segment glance cards ── */
  .seg-card {
    background: #1e2130; border: 1px solid #2d3148; border-radius: 14px;
    padding: 18px 20px;
  }
  .seg-card .seg-name { font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
  .seg-card .seg-count { font-size: 1.8rem; font-weight: 800; color: #e2e8f0; }
  .seg-card .seg-pct { font-size: 0.85rem; color: #64748b; }
  .seg-card .seg-desc { font-size: 0.8rem; color: #94a3b8; margin-top: 6px; line-height: 1.4; }

  /* ── Section page title ── */
  .page-title {
    font-size: 2rem; font-weight: 800; color: #e2e8f0; margin-bottom: 0.3rem;
  }
  .page-sub { color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem; }

  /* ── Recommendation cards ── */
  .rec-wrapper { display: flex; flex-direction: column; gap: 12px; margin-top: 1rem; }
  .rec-card {
    background: #1e2130; border: 1px solid #2d3148; border-radius: 14px;
    padding: 16px 20px; display: flex; align-items: center; gap: 16px;
  }
  .rec-rank {
    min-width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.95rem; color: white; flex-shrink: 0;
  }
  .rec-body { flex: 1; }
  .rec-name { font-weight: 700; color: #e2e8f0; font-size: 0.95rem; margin-bottom: 6px; }
  .rec-bar-bg {
    background: #2d3148; border-radius: 999px; height: 8px; margin-bottom: 4px;
  }
  .rec-bar-fill {
    height: 8px; border-radius: 999px;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
  }
  .rec-meta { font-size: 0.78rem; color: #64748b; }
  .rec-badge {
    background: #2d3148; border-radius: 999px;
    padding: 4px 12px; font-size: 0.8rem; font-weight: 700;
    color: #a78bfa; white-space: nowrap; flex-shrink: 0;
  }

  /* ── Match quality tags ── */
  .tag-high   { background:#14532d; color:#4ade80; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
  .tag-medium { background:#713f12; color:#fbbf24; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
  .tag-low    { background:#1e3a5f; color:#60a5fa; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:700; }

  /* ── Segment result card ── */
  .result-card {
    border-radius: 20px; padding: 28px 36px; margin: 1rem 0;
    text-align: center; color: white;
  }
  .result-card .result-emoji { font-size: 3rem; }
  .result-card .result-seg  { font-size: 2rem; font-weight: 900; margin: 6px 0; }
  .result-card .result-desc { font-size: 0.95rem; opacity: 0.88; }

  /* ── Action box ── */
  .action-box {
    background: #1e2130; border: 1px solid #2d3148; border-radius: 12px;
    padding: 16px 20px; margin-top: 1rem; color: #e2e8f0; font-size: 0.95rem;
  }
  .action-box strong { color: #a78bfa; }

  /* ── Compare bars ── */
  .compare-card {
    background: #1e2130; border: 1px solid #2d3148; border-radius: 14px;
    padding: 18px 20px;
  }
  .compare-label { color: #94a3b8; font-size: 0.85rem; margin-bottom: 6px; }
  .compare-value { color: #e2e8f0; font-size: 1.4rem; font-weight: 800; margin-bottom: 8px; }
  .compare-bar-bg { background: #2d3148; border-radius: 999px; height: 10px; }
  .compare-bar-fill { height: 10px; border-radius: 999px; }
  .compare-pct { font-size: 0.8rem; color: #64748b; margin-top: 6px; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(90deg, #a78bfa, #60a5fa) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 0.55rem 1.6rem !important; font-weight: 700 !important;
    font-size: 0.95rem !important; width: 100%;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  /* ── Selectbox / inputs dark theme ── */
  [data-testid="stSelectbox"] > div, [data-testid="stNumberInput"] > div {
    background: #1e2130 !important;
  }
  .stSlider > div { color: #94a3b8; }
  hr { border-color: #2d3148 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/kmeans_model.pkl", "rb") as f: kmeans = pickle.load(f)
    with open("models/scaler.pkl",       "rb") as f: scaler = pickle.load(f)
    with open("models/label_map.pkl",    "rb") as f: label_map = pickle.load(f)
    similarity_df = pd.read_pickle("models/similarity_df.pkl")
    rfm = pd.read_csv("models/rfm_segments.csv")
    return kmeans, scaler, label_map, similarity_df, rfm

kmeans, scaler, label_map, similarity_df, rfm = load_models()

# Clean product list — filter out anything that looks like a code (short / all-numeric)
product_list = sorted([
    p for p in similarity_df.index.tolist()
    if len(p) > 5 and not p.replace(" ", "").isdigit()
])

# ── Segment metadata ──────────────────────────────────────────────────────────
SEGMENT_META = {
    "High-Value": {
        "emoji": "💎", "color": "#c026d3",
        "gradient": "linear-gradient(135deg, #7e22ce, #c026d3)",
        "description": "Recent, frequent, and high-spending customers. Your most loyal buyers.",
        "action": "🎁 Offer exclusive loyalty rewards, VIP early access & personalised thank-you gifts.",
    },
    "Regular": {
        "emoji": "⭐", "color": "#2563eb",
        "gradient": "linear-gradient(135deg, #1d4ed8, #38bdf8)",
        "description": "Consistent mid-tier buyers with steady purchase behaviour.",
        "action": "📧 Send personalised upsell campaigns, bundle deals & limited-time offers.",
    },
    "Occasional": {
        "emoji": "🛍️", "color": "#16a34a",
        "gradient": "linear-gradient(135deg, #15803d, #4ade80)",
        "description": "Infrequent buyers with lower spend — high growth potential.",
        "action": "💌 Trigger re-engagement emails with seasonal discounts & first-purchase incentives.",
    },
    "At-Risk": {
        "emoji": "⚠️", "color": "#ea580c",
        "gradient": "linear-gradient(135deg, #c2410c, #fb923c)",
        "description": "Haven't purchased in a long time. High churn risk.",
        "action": "🆘 Launch win-back campaign: time-limited discount + personal outreach message.",
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
        similarity_df[pname].drop(pname)
        .sort_values(ascending=False).head(n).reset_index()
    )
    recs.columns = ["Product", "Similarity"]
    recs["Similarity"] = recs["Similarity"].round(4)
    return recs, pname

def predict_segment(recency, frequency, monetary):
    x = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans.predict(x)[0]
    return label_map[cluster], cluster

def match_tag(sim):
    pct = sim * 100
    if pct >= 60:   return '<span class="tag-high">🔥 Strong Match</span>'
    elif pct >= 30: return '<span class="tag-medium">✨ Good Match</span>'
    else:           return '<span class="tag-low">💡 Related</span>'

# ── Session state for navigation ─────────────────────────────────────────────
if "nav" not in st.session_state:
    st.session_state.nav = "🏠 Home"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛒 Shopper Spectrum</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🎯 Product Recommendations", "👥 Customer Segmentation"],
        index=["🏠 Home", "🎯 Product Recommendations", "👥 Customer Segmentation"].index(st.session_state.nav),
        label_visibility="collapsed",
    )
    st.session_state.nav = page
    st.markdown("---")
    st.markdown("**📊 Dataset Stats**")
    st.metric("Customers Segmented", f"{len(rfm):,}")
    st.metric("Products in Catalogue", f"{len(similarity_df):,}")
    st.metric("Clusters (k)", "4")
    st.markdown("---")
    st.markdown("<small style='color:#475569'>Built with KMeans + Cosine Similarity</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.nav == "🏠 Home":
    st.markdown('<div class="hero-title">🛒 Shopper Spectrum</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Customer Segmentation & Product Recommendations for E-Commerce</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="feature-card">
          <h3>🎯 Product Recommendations</h3>
          <p>Enter any product name and instantly get the <strong>top similar products</strong>
          based on real customer purchase patterns — powered by item-based collaborative
          filtering with cosine similarity.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("Go to Recommendations ➜", key="btn_rec"):
            st.session_state.nav = "🎯 Product Recommendations"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="feature-card">
          <h3>👥 Customer Segmentation</h3>
          <p>Enter a customer's <strong>Recency, Frequency, and Monetary</strong> values
          to instantly predict their segment: High-Value, Regular, Occasional, or At-Risk —
          using a trained KMeans model.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("Go to Segmentation ➜", key="btn_seg"):
            st.session_state.nav = "👥 Customer Segmentation"
            st.rerun()

    st.markdown("---")
    st.markdown("### 🏷️ Customer Segments at a Glance")
    cols = st.columns(4, gap="medium")
    for col, (seg, meta) in zip(cols, SEGMENT_META.items()):
        with col:
            seg_count = int((rfm["Segment"] == seg).sum())
            pct = seg_count / len(rfm) * 100
            st.markdown(f"""
            <div class="seg-card">
              <div class="seg-name" style="color:{meta['color']}">{meta['emoji']} {seg}</div>
              <div class="seg-count">{seg_count:,}</div>
              <div class="seg-pct">{pct:.1f}% of customers</div>
              <div class="seg-desc">{meta['description']}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.nav == "🎯 Product Recommendations":
    st.markdown('<div class="page-title">🎯 Product Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Discover similar products based on real customer purchase behaviour</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1], gap="medium")
    with col1:
        product_input = st.selectbox(
            "🔍 Select a product:",
            options=["— Select a product —"] + product_list,
            index=0,
        )
    with col2:
        n_recs = st.slider("# Recommendations", min_value=3, max_value=10, value=5)

    if st.button("🚀 Get Recommendations", key="btn_get_rec"):
        if product_input == "— Select a product —":
            st.warning("⚠️ Please select a product first.")
        else:
            recs, matched = get_recommendations(product_input, n=n_recs)
            if recs is None:
                st.error(f"❌ Product not found. Try a different name.")
            else:
                if matched != product_input.strip().upper():
                    st.info(f"ℹ️ Matched to: **{matched}**")

                # ── Selected product banner ──────────────────────────────────
                st.markdown(f"""
                <div style="background:#1e2130;border:1px solid #a78bfa;border-radius:14px;
                            padding:16px 22px;margin:1rem 0;">
                  <div style="color:#94a3b8;font-size:0.82rem;margin-bottom:4px;">SELECTED PRODUCT</div>
                  <div style="color:#e2e8f0;font-size:1.15rem;font-weight:800;">🛒 {matched}</div>
                  <div style="color:#64748b;font-size:0.82rem;margin-top:4px;">
                    Showing top {n_recs} most similar products
                  </div>
                </div>""", unsafe_allow_html=True)

                # ── Recommendation cards ─────────────────────────────────────
                max_sim = recs["Similarity"].max() if recs["Similarity"].max() > 0 else 1
                cards_html = '<div class="rec-wrapper">'
                for i, row in recs.iterrows():
                    pct_raw   = row["Similarity"] * 100          # raw cosine %
                    bar_pct   = (row["Similarity"] / max_sim) * 100  # relative bar width
                    tag       = match_tag(row["Similarity"])
                    cards_html += f"""
                    <div class="rec-card">
                      <div class="rec-rank">#{i+1}</div>
                      <div class="rec-body">
                        <div class="rec-name">{row['Product']}</div>
                        <div class="rec-bar-bg">
                          <div class="rec-bar-fill" style="width:{bar_pct:.1f}%"></div>
                        </div>
                        <div class="rec-meta">
                          {tag}&nbsp;&nbsp;
                          <span style="color:#94a3b8">{pct_raw:.1f}% match score</span>
                        </div>
                      </div>
                      <div class="rec-badge">{pct_raw:.1f}%</div>
                    </div>"""
                cards_html += '</div>'
                st.markdown(cards_html, unsafe_allow_html=True)

                # ── Summary table ────────────────────────────────────────────
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                with st.expander("📋 View as Table"):
                    display_df = recs.copy()
                    display_df["Match %"] = (display_df["Similarity"] * 100).round(1).astype(str) + "%"
                    display_df["Quality"] = display_df["Similarity"].apply(
                        lambda s: "🔥 Strong" if s >= 0.6 else ("✨ Good" if s >= 0.3 else "💡 Related")
                    )
                    st.dataframe(
                        display_df[["Product", "Match %", "Quality"]].rename(
                            columns={"Product": "Recommended Product"}
                        ),
                        use_container_width=True, hide_index=True
                    )


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.nav == "👥 Customer Segmentation":
    st.markdown('<div class="page-title">👥 Customer Segmentation Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Enter RFM values to instantly predict the customer segment</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        recency = st.number_input(
            "📅 Recency — days since last purchase",
            min_value=0, max_value=730, value=30, step=1,
            help="Lower = more recent. 0 = bought today."
        )
    with col2:
        frequency = st.number_input(
            "🔁 Frequency — number of orders",
            min_value=1, max_value=500, value=10, step=1,
            help="Total distinct invoices for this customer."
        )
    with col3:
        monetary = st.number_input(
            "💰 Monetary — total spend (£)",
            min_value=1.0, max_value=100000.0, value=500.0, step=10.0,
            help="Total amount spent by the customer."
        )

    if st.button("🔮 Predict Segment", key="btn_predict"):
        segment, cluster = predict_segment(recency, frequency, monetary)
        meta = SEGMENT_META[segment]

        # ── Result banner ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card" style="background:{meta['gradient']}">
          <div class="result-emoji">{meta['emoji']}</div>
          <div class="result-seg">{segment} Customer</div>
          <div class="result-desc">{meta['description']}</div>
        </div>""", unsafe_allow_html=True)

        # ── Input summary ────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        m1.metric("📅 Recency",   f"{recency} days")
        m2.metric("🔁 Frequency", f"{frequency} orders")
        m3.metric("💰 Monetary",  f"£{monetary:,.2f}")
        m4.metric("🏷️ Cluster",   f"#{cluster}")

        # ── Recommended action ───────────────────────────────────────────────
        st.markdown(f"""
        <div class="action-box">
          <strong>💡 Recommended Action</strong><br/><br/>
          {meta['action']}
        </div>""", unsafe_allow_html=True)

        # ── How this customer compares ────────────────────────────────────────
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown("### 📊 How This Customer Compares")

        compare_cols = st.columns(3, gap="medium")
        compare_fields = [
            ("Recency",   recency,   "days",   "#a78bfa", False),   # lower is better
            ("Frequency", frequency, "orders", "#60a5fa", True),
            ("Monetary",  monetary,  "£",      "#4ade80", True),
        ]

        for col, (field, val, unit, color, higher_better) in zip(compare_cols, compare_fields):
            pct_below = int((rfm[field] <= val).mean() * 100)
            if higher_better:
                bar_pct   = pct_below
                label_txt = f"Better than {pct_below}% of customers"
            else:
                bar_pct   = 100 - pct_below
                label_txt = f"More recent than {100 - pct_below}% of customers"

            val_display = f"£{val:,.0f}" if unit == "£" else f"{val:,} {unit}"
            with col:
                st.markdown(f"""
                <div class="compare-card">
                  <div class="compare-label">{field}</div>
                  <div class="compare-value">{val_display}</div>
                  <div class="compare-bar-bg">
                    <div class="compare-bar-fill"
                         style="width:{bar_pct}%; background:{color}"></div>
                  </div>
                  <div class="compare-pct">{label_txt}</div>
                </div>""", unsafe_allow_html=True)
