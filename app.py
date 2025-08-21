import streamlit as st
import pandas as pd
import numpy as np
import io
import string
import nltk
from nltk.corpus import stopwords
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------
# Download stopwords
# ----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------------------------
# Text Preprocessing
# ------------------------------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

# ------------------------------------
# Month to Quarter Conversion
# ------------------------------------
def get_quarter_month(month):
    if month in [1, 2, 3]:
        return 'Q1', 'Jan-Feb-Mar'
    elif month in [4, 5, 6]:
        return 'Q2', 'Apr-May-Jun'
    elif month in [7, 8, 9]:
        return 'Q3', 'Jul-Aug-Sep'
    else:
        return 'Q4', 'Oct-Nov-Dec'

# ------------------------------------
# BERTopic Analysis (for better grouping)
# ------------------------------------
def analyze_tickets_bertopic(df, date_col, subject_col):
    # Rename columns
    df = df.rename(columns={date_col: 'DateTime', subject_col: 'Subject'})

    # Convert Date column
    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DateTime'])

    # Preprocess text
    df['CleanSubject'] = df['Subject'].apply(preprocess_text)

    # Extract time details
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df[['Quarter', 'Period']] = df.apply(lambda r: pd.Series(get_quarter_month(r['Month'])), axis=1)
    df['QuarterYear'] = df['Quarter'] + ' ' + df['Year'].astype(str)

    # BERTopic modeling (kept for topic grouping internally, but not displayed)
    vectorizer_model = CountVectorizer(stop_words='english')
    topic_model = BERTopic(vectorizer_model=vectorizer_model, language="english")
    topics, probs = topic_model.fit_transform(df['CleanSubject'])
    df['Topic'] = topics

    return df

# ------------------------------------
# Summarize Quarterly Issues (Actual Subjects)
# ------------------------------------
def summarize_quarterly_tickets(df):
    summarized = []

    grouped = df.groupby(['QuarterYear', 'Period'])

    for (quarter, period), group in grouped:
        total_tickets = group.shape[0]

        # Most frequent actual ticket subjects
        top_issues_counts = group['Subject'].value_counts().head(3)
        most_frequent_issue = top_issues_counts.index[0] if not top_issues_counts.empty else "N/A"
        common_issues = '; '.join(top_issues_counts.index.tolist())

        summarized.append({
            'Quarter': quarter,
            'Period': period,
            'Total Tickets': total_tickets,
            'Most Frequent Issue': most_frequent_issue,
            'Common Issues': common_issues
        })

    return pd.DataFrame(summarized)

# ------------------------------------
# Streamlit UI
# ------------------------------------
st.title("Quarterly Ticket Summary")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        cols = df.columns.tolist()
        date_col = st.selectbox("Select the Date/Time column", cols)
        subject_col = st.selectbox("Select the Issue Header column", cols)

        if st.button("Generate Quarterly Summary"):
            with st.spinner("Processing tickets..."):
                # Preprocess + Quarter extraction
                result_df = analyze_tickets_bertopic(df, date_col, subject_col)

                # Summarized quarterly issues based on actual subjects
                final_summary_df = summarize_quarterly_tickets(result_df)

                # Display summarized quarterly issues
                st.subheader("Quarterly Summary (Actual Issues)")
                st.dataframe(final_summary_df)

                # Download summarized Excel
                towrite = io.BytesIO()
                final_summary_df.to_excel(towrite, index=False, engine='openpyxl')
                towrite.seek(0)
                st.download_button(
                    label="Download Quarterly Summary",
                    data=towrite,
                    file_name="quarterly_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

else:
    st.info("ℹ️ Please upload an Excel file to begin analysis.")
