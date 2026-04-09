import streamlit as st
import requests
import random
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Search Ranking", layout="wide")

st.title("🔍 AI Search Ranking System")

# ── Sidebar ─────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
user_id = st.sidebar.text_input("User ID", "user_1")
experiment = st.sidebar.text_input("Experiment Name", "exp_ltr_v2")
num_candidates = st.sidebar.slider("Number of candidates", 1, 10, 5)

# ── Query Input ─────────────────────────────────────────
query = st.text_input("Enter search query", "machine learning")

# ── Candidate Generator (FIXED VERSION) ─────────────────
def generate_candidates(n=5):
    candidates = []
    for i in range(n):
        candidates.append({
            "doc_id": f"doc_{i}",
            "bm25_score": random.uniform(5, 20),
            "tfidf_cosine": random.uniform(0, 1),
            "query_term_coverage": random.uniform(0, 1),
            "title_match_score": random.uniform(0, 1),
            "body_match_score": random.uniform(0, 1),
            "doc_length": random.uniform(100, 2000),
            "doc_pagerank": random.uniform(0, 1),
            "doc_freshness_days": random.uniform(1, 365),
            "avg_click_rate": random.uniform(0, 0.5),
            "query_length": random.randint(1, 5),
            "query_idf_sum": random.uniform(5, 50),
            "is_navigational": random.choice([0, 1])
        })
    return candidates


# ── Session state ───────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Search Button ───────────────────────────────────────
if st.button("🔍 Search"):
    payload = {
        "query": query,
        "user_id": user_id,
        "experiment": experiment,
        "candidates": generate_candidates(num_candidates)
    }

    res = requests.post(f"{API_URL}/rank", json=payload)

    if res.status_code == 200:
        st.session_state.results = res.json()
    else:
        st.error("API Error")
        st.write(res.text)

# ── Display Results ─────────────────────────────────────
if st.session_state.results:
    data = st.session_state.results

    st.subheader("📄 Results")

    for r in data["results"]:
        col1, col2, col3 = st.columns([4, 2, 2])

        with col1:
            st.write(f"**{r['doc_id']}**")

        with col2:
            st.write(f"Rank: {r['rank']}")

        with col3:
            st.write(f"Score: {round(r['score'], 4)}")

        # ── Click Button ─────────────────────────────
        if st.button(f"Click {r['doc_id']}", key=f"click_{r['doc_id']}"):
            requests.post(f"{API_URL}/event/click", json={
                "user_id": user_id,
                "experiment": data["experiment"],
                "variant": data["variant"],
                "query": query,
                "doc_id": r["doc_id"],
                "rank": r["rank"]
            })
            st.success(f"Clicked {r['doc_id']}")

        st.divider()

    # ── Conversion Button ───────────────────────────
    if st.button("💰 Convert (Top Result)"):
        top_doc = data["results"][0]["doc_id"]

        requests.post(f"{API_URL}/event/conversion", json={
            "user_id": user_id,
            "experiment": data["experiment"],
            "variant": data["variant"],
            "query": query,
            "doc_id": top_doc
        })

        st.success(f"Conversion recorded for {top_doc}")

    # ── Experiment Info ─────────────────────────────
    st.success(f"Experiment: {data['experiment']} | Variant: {data['variant']}")

# ── A/B Report Section ─────────────────────────────────
st.subheader("📊 A/B Experiment Dashboard")

if st.button("Load Report"):
    res = requests.get(f"{API_URL}/ab/report/{experiment}")

    if res.status_code == 200:
        report = res.json()

        # ── Variant Metrics ────────────────────────
        st.write("### Variant Metrics")

        df_variants = pd.DataFrame(report["variants"]).T
        st.dataframe(df_variants, use_container_width=True)

        # ── Comparisons ───────────────────────────
        st.write("### Comparisons")

        df_comp = pd.DataFrame(report["comparisons"])
        st.dataframe(df_comp, use_container_width=True)

        # ── Visualization ─────────────────────────
        if not df_variants.empty:
            st.write("### CTR Comparison")
            st.bar_chart(df_variants["ctr"])

    else:
        st.error("Failed to load report")