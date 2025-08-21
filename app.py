import streamlit as st
import pandas as pd
import numpy as np
import io
import string
import nltk
from nltk.corpus import stopwords
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

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
# BERTopic Analysis
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

    # BERTopic modeling
    vectorizer_model = CountVectorizer(stop_words='english')
    topic_model = BERTopic(vectorizer_model=vectorizer_model, language="english")
    topics, probs = topic_model.fit_transform(df['CleanSubject'])
    df['Topic'] = topics

    # Convert topic numbers to human-readable names
    def topic_to_name(topic, top_n=5):
        if topic == -1:
            return "Other / Miscellaneous"
        words = topic_model.get_topic(topic)
        if words:
            return ', '.join([w[0] for w in words[:top_n]])
        else:
            return "Other / Miscellaneous"

    df['TopicName'] = df['Topic'].apply(lambda t: topic_to_name(t, top_n=5))

    # Group by Quarter + Topic
    grouped = df.groupby(['QuarterYear', 'Period', 'TopicName'])
    summary = []
    for (qyear, period, topic_name), group in grouped:
        summary.append({
            'Quarter': qyear,
            'Period': period,
            'Topic': topic_name,
            'No. of Tickets': group.shape[0]
        })
    summary_df = pd.DataFrame(summary)

    # Fill missing topic combinations with zero counts
    years = df['Year'].unique()
    quarter_order = ['Q1', 'Q2', 'Q3', 'Q4']
    periods_map = {
        'Q1': 'Jan-Feb-Mar',
        'Q2': 'Apr-May-Jun',
        'Q3': 'Jul-Aug-Sep',
        'Q4': 'Oct-Nov-Dec'
    }

    all_rows = []
    for year in sorted(years):
        for q in quarter_order:
            period_name = periods_map[q]
            for topic in summary_df['Topic'].unique():
                all_rows.append({'Quarter': f"{q} {year}", 'Period': period_name, 'Topic': topic})

    all_df = pd.DataFrame(all_rows)
    merged = pd.merge(all_df, summary_df, on=['Quarter', 'Period', 'Topic'], how='left')
    merged['No. of Tickets'] = merged['No. of Tickets'].fillna(0).astype(int)

    # Sort properly
    merged['Year'] = merged['Quarter'].str.split().str[1].astype(int)
    merged['Qnum'] = merged['Quarter'].str.split().str[0].apply(lambda x: quarter_order.index(x) + 1)
    merged = merged.sort_values(by=['Year', 'Qnum', 'Topic']).reset_index(drop=True)
    merged.drop(columns=['Year', 'Qnum'], inplace=True)

    return merged, topic_model, df

# ------------------------------------
# Summarize Quarterly Issues (using actual subjects)
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
st.title("📊 Ticket Analysis with BERTopic NLP + Actual Issue Summarization")

uploaded_file = st.file_uploader("📂 Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(df.head())

        cols = df.columns.tolist()
        date_col = st.selectbox("📅 Select the Date/Time column", cols)
        subject_col = st.selectbox("📝 Select the Issue Header column", cols)

        if st.button("🚀 Analyze Tickets"):
            with st.spinner("🔄 Analyzing topics..."):
                summary_df, model, result_df = analyze_tickets_bertopic(df, date_col, subject_col)

                # Detailed BERTopic summary
                st.subheader("📊 Detailed Quarterly Topic Summary")
                st.dataframe(summary_df)

                # Download detailed BERTopic summary
                towrite = io.BytesIO()
                summary_df.to_excel(towrite, index=False, engine='openpyxl')
                towrite.seek(0)
                st.download_button(
                    label="📥 Download Detailed Topic Summary",
                    data=towrite,
                    file_name="ticket_summary_bertopic.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Summarized quarterly issues based on actual subjects
                final_summary_df = summarize_quarterly_tickets(result_df)

                st.subheader("📌 Quarterly Summary (Actual Issues)")
                st.dataframe(final_summary_df)

                # Download summarized Excel
                towrite2 = io.BytesIO()
                final_summary_df.to_excel(towrite2, index=False, engine='openpyxl')
                towrite2.seek(0)
                st.download_button(
                    label="📥 Download Quarterly Issue Summary",
                    data=towrite2,
                    file_name="quarterly_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Top Topics & Keywords
                st.markdown("---")
                st.subheader("🔑 Top Topics and Their Keywords")
                topic_info = model.get_topic_info().head(10)
                for topic_num in topic_info['Topic']:
                    if topic_num == -1:
                        continue
                    topic_words = model.get_topic(topic_num)
                    word_list = ', '.join([f"{w[0]} ({w[1]:.2f})" for w in topic_words[:10]])
                    st.markdown(f"**Topic {topic_num}:** {word_list}")

                # UMAP Visualization
                st.markdown("---")
                st.subheader("🌐 UMAP Topic Visualization")
                fig = model.visualize_topics()
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

else:
    st.info("ℹ️ Please upload an Excel file to begin analysis.")
