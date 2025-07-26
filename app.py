# --- Suppress warnings only ---
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import joblib
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import preprocess_text, predict_status, predict_claim_type, predict_score, get_guidance

# --- Load Dataset ---
@st.cache_data
def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "synthetic_petitions_dataset.csv")
    return pd.read_csv(path)

# --- Extract Steps from Text ---
def extract_action_steps(guidance_text: str):
    return [line.strip("📌 ").strip() for line in guidance_text.strip().split("\n") if line.strip()]

# --- Dynamic Vertical Sankey Diagram ---
def plot_plan_of_action_graph(steps):
    if len(steps) < 2:
        st.info("Not enough action steps to draw a diagram.")
        return

    node_labels = steps
    source = list(range(len(steps) - 1))
    target = list(range(1, len(steps)))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        orientation="v",
        node=dict(
            pad=30,
            thickness=30,
            line=dict(color="gray", width=1),
            label=node_labels,
            color="cornflowerblue",
            hovertemplate='%{label}<extra></extra>',
        ),
        link=dict(
            source=source,
            target=target,
            value=[1] * len(source),
            color=["lightgray"] * len(source)
        )
    )])

    fig.update_layout(
        title="🔗 Plan of Action Flow (Police + AI Collaboration)",
        font=dict(size=14, color='black'),
        height=100 * len(steps),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Show GIFs with Sankey Diagram ---
def display_flow_with_gifs(steps):
    col1, col2, col3 = st.columns([1, 5, 1])

    with col1:
        if os.path.exists("police.gif"):
            st.image("police.gif", width=150, caption="👮‍♂️ Police")
            st.markdown("<h2 style='text-align:center;'>➡️</h2>", unsafe_allow_html=True)

    with col2:
        plot_plan_of_action_graph(steps)

    with col3:
        if os.path.exists("ai.gif"):
            st.markdown("<h2 style='text-align:center;'>⬅️</h2>", unsafe_allow_html=True)
            st.image("ai.gif", width=200, caption="🤖 AI System")

# --- Main Use Case Renderer ---
def draw_use_case_diagram_dynamic(guidance_text):
    steps = extract_action_steps(guidance_text)
    st.subheader("📘 Dynamic Use Case Diagram")
    display_flow_with_gifs(steps)

# --- Show Action Plan as Text ---
def generate_plan_of_action(guidance_text):
    st.subheader("🧭 Plan of Action")
    for line in guidance_text.strip().split("\n"):
        st.markdown(f"📌 {line}")

# --- UI Config ---
st.set_page_config(layout="wide")

# --- Display Character GIFs: Police | Sound | AI ---
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    st.image("police.gif", use_column_width=True)
with colB:
    st.image("sound 2.gif", use_column_width=True)
with colC:
    st.image("ai.gif", use_column_width=True)

# --- Headline ---
st.markdown("<h1 style='text-align: center; color: white;'>AI-Based Grievance Redressal System</h1>", unsafe_allow_html=True)

# --- Sidebar Navigation ---
page = st.sidebar.radio("📂 Select View", ["📍 Live Case Assistant", "📊 Dataset Case Insights"])

# --- PAGE 1: LIVE CASE ANALYSIS ---
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
                draw_use_case_diagram_dynamic(guidance_text)
                st.markdown("##### 🧭 This flowchart represents the AI + Police recommended course of action based on the case inputs.")

            # Dashboard
            df = pd.DataFrame(st.session_state.cases)

            st.subheader("📊 Case Analysis Dashboard")
            with st.expander("ℹ️ What does this dashboard show?"):
                st.markdown("""
                The **Case Analysis Dashboard** gives a quick visual overview of all analyzed cases so far:

                - **📘 Case Outcome Distribution (Pie Chart):** Closed vs Running
                - **📚 Type of Cases Handled (Bar Chart):** Complaint type distribution
                - **📉 Severity of Case Scores (Histogram)**
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

                st.markdown("### 🗂️ Analyzed Case Records")
                st.dataframe(df[::-1], use_container_width=True)

# --- PAGE 2: DATASET INSIGHTS ---
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
        This dashboard shows analytics over all historical/past cases in the dataset:

        - 📘 Closed vs Running outcomes
        - 📚 Complaint type frequency
        - 📉 Case score (severity) spread
        - 🗂️ Full dataset view
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

            st.markdown("### 🗂️ Filtered Dataset Records")
            st.dataframe(df_filtered[::-1], use_container_width=True)
