import pandas as pd
import nltk
import io
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import hdbscan
from collections import Counter
import streamlit as st

# ----------------------------
# 1. Setup
# ----------------------------
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------
# 2. Utility Functions
# ----------------------------
def clean_text(text):
    words = [w for w in str(text).lower().split() if w not in STOPWORDS]
    return " ".join(words)

def extract_quarter(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["quarter"] = df[date_col].dt.to_period("Q").astype(str)
    return df

def analyze_issues(df, date_col, issue_col):
    # Create quarter column from datetime
    df = extract_quarter(df, date_col)

    # Drop rows where issues are missing
    df = df.dropna(subset=[issue_col])

    # Clean issue texts
    df["clean_issue"] = df[issue_col].apply(clean_text)

    # Generate embeddings
    embeddings = model.encode(df["clean_issue"].tolist(), show_progress_bar=False)

    # Cluster similar issues using HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    df["cluster"] = clusterer.fit_predict(embeddings)
    df["cluster"] = df["cluster"].apply(lambda x: -1 if x == -1 else x)

    # Get representative issue per cluster
    def get_representative_issue(cluster_df):
        return Counter(cluster_df[issue_col]).most_common(1)[0][0]

    cluster_labels = df[df["cluster"] != -1].groupby("cluster").apply(get_representative_issue)
    df["cluster_label"] = df["cluster"].map(cluster_labels).fillna(df[issue_col])

    # Summarize issues per quarter
    summary = df.groupby(["quarter", "cluster_label"]).size().reset_index(name="Count")

    def summarize_issues(group):
        sorted_group = group.sort_values("Count", ascending=False)
        common_issues = ", ".join(f"{row['cluster_label']}({row['Count']})" for _, row in sorted_group.iterrows())
        top_issues = ", ".join(sorted_group.head(2)["cluster_label"])
        total_count = sorted_group["Count"].sum()
        return pd.Series({
            "frequency": total_count,
            "common issue": common_issues,
            "common frequent issue": top_issues
        })

    summary_df = summary.groupby("quarter").apply(summarize_issues).reset_index()
    return summary_df

# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Quarterly Issue Analyzer", layout="wide")
st.title("AI-Powered Quarterly Issue Analyzer")
st.markdown("Upload your **Excel file** with a datetime column and get an **AI-driven quarterly summary** of common issues.")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    # Load Excel
    df = pd.read_excel(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Let the user select datetime and issue columns
    st.markdown("### Select Columns")
    date_col = st.selectbox("Select Date Column", df.columns)
    issue_col = st.selectbox("Select Issue Column", df.columns)

    if st.button("Analyze Issues"):
        with st.spinner("Analyzing issues using AI... ⏳"):
            result_df = analyze_issues(df, date_col, issue_col)

        st.success("✅ Analysis complete!")
        st.subheader("📌 AI-Generated Quarterly Summary")
        st.dataframe(result_df, use_container_width=True)

        # Download analyzed Excel
        output = io.BytesIO()
        result_df.to_excel(output, index=False)
        output.seek(0)
        st.download_button(
            label="Download Cleaned Summary Excel",
            data=output,
            file_name="quarterly_issue_summary_ai.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
