import streamlit as st
import joblib
import os
from utils import preprocess_text, predict_status, predict_claim_type, predict_score, get_guidance
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import plotly.express as px
import pandas as pd

@st.cache_data
def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "synthetic_petitions_dataset.csv")
    return pd.read_csv(path)

def generate_plan_of_action(guidance_text):
    st.subheader("🧭 Plan of Action")
    for line in guidance_text.strip().split("\n"):
        st.markdown(f"📌 {line}")

def draw_use_case_diagram():
    # Load PNGs converted from GIFs
    police_icon_path = os.path.join(os.path.dirname(__file__), "police_icon.png")
    ai_icon_path = os.path.join(os.path.dirname(__file__), "ai_icon.png")
    police_img = plt.imread(police_icon_path)
    ai_img = plt.imread(ai_icon_path)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 700)
    ax.axis('off')

    # Background lines
    for y in range(0, 700, 20):
        ax.add_line(plt.Line2D((0, 1000), (y, y), lw=0.5, color='lightblue'))

    # Actor labels
    ax.text(100, 620, "👮\nPolice Officer", fontsize=12, ha='center', fontname="Comic Sans MS")
    ax.text(900, 620, "🤖\nAI System", fontsize=12, ha='center', fontname="Comic Sans MS")

    # Actor icons (police & AI system)
    ax.imshow(police_img, extent=[30, 170, 630, 680], aspect='auto')
    ax.imshow(ai_img, extent=[830, 970, 630, 680], aspect='auto')

    # Use cases
    use_cases = [
        ("Upload Petition", 500, 550),
        ("Upload Response", 500, 480),
        ("Predict Case Status", 500, 410),
        ("Generate Action Plan", 500, 340),
        ("Show Dashboard", 500, 270)
    ]

    for text, x, y in use_cases:
        ellipse = patches.Ellipse((x, y), 240, 50, fill=True, color='lightgray', ec='black')
        ax.add_patch(ellipse)
        ax.text(x, y, text, fontsize=10, ha='center', va='center', fontname="Comic Sans MS")

        # Arrows from actors to use cases
        ax.plot([150, x - 130], [655, y], 'k-', lw=1.2)
        ax.plot([850, x + 130], [655, y], 'k-', lw=1.2)

    # Show diagram
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    st.subheader("📘 Use Case Diagram")
    st.image(buf)

# ------------------------------- #
#         STREAMLIT UI            #
# ------------------------------- #
st.set_page_config(layout="wide")
page = st.sidebar.radio("📂 Select View", ["📍 Live Case Assistant", "📊 Dataset Case Insights"])

# ------------------------------- #
# PAGE 1: LIVE CASE ASSISTANT     #
# ------------------------------- #
if page == "📍 Live Case Assistant":
    st.title("👮‍♂️ AI Police Assistance")

    petition = st.text_area("📄 Enter Petition Text")
    response = st.text_area("📝 Enter Investigating Officer's Response")

    if st.button("🔍 Analyze Case"):
        if not petition or not response:
            st.error("Please enter both petition and response.")
        else:
            claim = predict_claim_type(petition)
            score = predict_score(response)
            status = predict_status(petition, claim, response, score)

            st.write(f"### 🧾 Case Status: `{status}`")

            if 'cases' not in st.session_state:
                st.session_state.cases = []

            st.session_state.cases.append({
                "Petition": petition,
                "Claim": claim,
                "Response": response,
                "Score": score,
                "Status": status
            })

            if status.lower() == 'closed':
                st.success("✅ The case is closed. No further action required.")
            else:
                st.warning("🚨 The case is still running. Suggesting next steps...")
                guidance_text = get_guidance(petition, claim, response)
                generate_plan_of_action(guidance_text)
                draw_use_case_diagram()

            df = pd.DataFrame(st.session_state.cases)

            st.subheader("📊 Case Analysis Dashboard")
            with st.expander("ℹ️ What does this dashboard show?"):
                st.markdown("""
                The **Case Analysis Dashboard** gives a quick visual overview of all analyzed cases so far:

                - **📘 Case Outcome Distribution (Pie Chart):** Shows the ratio of `Closed` vs `Running` cases.
                - **📚 Type of Cases Handled (Bar Chart):** Frequency of complaint types.
                - **📉 Case Scores (Histogram):** Severity distribution.
                - **🗂️ Records Table:** All analyzed case details.
                """)

            if not df.empty:
                col1, col2, col3 = st.columns([1, 1, 1])

                status_counts = df['Status'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                fig1 = px.pie(status_counts, names='Status', values='Count', title='📘 Case Outcome Distribution')
                col1.plotly_chart(fig1, use_container_width=True)

                claim_counts = df['Claim'].value_counts().reset_index()
                claim_counts.columns = ['Claim', 'Count']
                fig2 = px.bar(claim_counts, x='Claim', y='Count', title='📚 Type of Cases Handled', color='Claim', text='Count')
                fig2.update_traces(textposition='outside')
                col2.plotly_chart(fig2, use_container_width=True)

                fig3 = px.histogram(df, x='Score', nbins=10, title='📉 Distribution of Case Scores', color_discrete_sequence=['orange'])
                col3.plotly_chart(fig3, use_container_width=True)

                st.markdown("---")
                st.markdown("### 🗂️ Analyzed Case Records")
                st.dataframe(df[::-1], use_container_width=True)

# ------------------------------- #
# PAGE 2: DATASET CASE INSIGHTS   #
# ------------------------------- #
elif page == "📊 Dataset Case Insights":
    st.title("📊 Dataset Case Insights Dashboard")
    df = load_dataset()
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        'complaint_text': 'Petition',
        'extracted_claims': 'Claim',
        'io_response': 'Response',
        'response_score': 'Score',
        'final_status': 'Status'
    })

    expected_cols = {'Petition', 'Claim', 'Response', 'Score', 'Status'}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        st.error(f"❌ Missing expected columns: {missing_cols}. Found columns: {df.columns.tolist()}")
        st.stop()

    with st.expander("ℹ️ What is this dashboard showing?"):
        st.markdown("""
        This dashboard shows analytics over all historical/past cases in the dataset.

        - **📘 Case Outcome Distribution:** Closed vs Running over all cases
        - **📚 Case Type Distribution:** Frequency of each type of case
        - **📉 Score Distribution:** Severity distribution across all records
        - **🗂️ Case Log Table:** Full listing of dataset records
        """)

    if not df.empty:
        st.markdown("### 🔍 Filter Options")
        col1, col2, col3 = st.columns(3)

        status_filter = col1.multiselect("Filter by Status", options=sorted(df['Status'].unique()), default=sorted(df['Status'].unique()))
        claim_filter = col2.multiselect("Filter by Claim Type", options=sorted(df['Claim'].unique()), default=sorted(df['Claim'].unique()))
        score_range = col3.slider("Filter by Score Range", min_value=int(df['Score'].min()), max_value=int(df['Score'].max()), value=(int(df['Score'].min()), int(df['Score'].max())))

        df_filtered = df[
            (df['Status'].isin(status_filter)) &
            (df['Claim'].isin(claim_filter)) &
            (df['Score'] >= score_range[0]) &
            (df['Score'] <= score_range[1])
        ]

        st.markdown("---")

        if df_filtered.empty:
            st.warning("No records match your filter criteria.")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])

            status_counts = df_filtered['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig1 = px.pie(status_counts, names='Status', values='Count', title='📘 Dataset Outcome Distribution')
            col1.plotly_chart(fig1, use_container_width=True)

            claim_counts = df_filtered['Claim'].value_counts().reset_index()
            claim_counts.columns = ['Claim', 'Count']
            fig2 = px.bar(claim_counts, x='Claim', y='Count', title='📚 Case Type Frequencies', color='Claim', text='Count')
            fig2.update_traces(textposition='outside')
            col2.plotly_chart(fig2, use_container_width=True)

            fig3 = px.histogram(df_filtered, x='Score', nbins=10, title='📉 Case Score Severity Spread', color_discrete_sequence=['teal'])
            col3.plotly_chart(fig3, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🗂️ Filtered Dataset Records")
            st.dataframe(df_filtered[::-1], use_container_width=True)
