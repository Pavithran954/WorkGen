import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from autoviz.AutoViz_Class import AutoViz_Class
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from datetime import datetime

nltk.download('punkt')

st.set_page_config(page_title="WorkGEN", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "df" not in st.session_state:
    st.session_state.df = None
if "report_content" not in st.session_state:
    st.session_state.report_content = []
if "generated_charts" not in st.session_state:
    st.session_state.generated_charts = set()
if "autoviz_run" not in st.session_state:
    st.session_state.autoviz_run = False
if "projects" not in st.session_state:  
    st.session_state.projects = {}

def rename_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def is_project_exists(project_name):
    return project_name in st.session_state.projects

def add_project(project_name, num_members, df):
    # Check for EmpID or EmpName and JobSatisfaction or PerformanceLevel
    emp_col = None
    job_satisfaction_col = None
    
    # Determine employee identifier column
    if 'EmpID' in df.columns:
        emp_col = 'EmpID'
    elif 'EmpName' in df.columns:
        emp_col = 'EmpName'
    else:
        st.error("The dataset should contain either 'EmpID' or 'EmpName' for creating project credentials.")
        return False
    
    # Determine performance measure column
    if 'JobSatisfaction' in df.columns:
        job_satisfaction_col = 'JobSatisfaction'
    elif 'PerformanceLevel' in df.columns:
        job_satisfaction_col = 'PerformanceLevel'
    else:
        st.error("The dataset should contain either 'JobSatisfaction' or 'PerformanceLevel' for creating project credentials.")
        return False

    # Check if project already exists
    if is_project_exists(project_name):
        st.error(f"Project '{project_name}' already exists.")
        return False

    # Filter eligible employees based on performance measure
    eligible_employees = df[df[job_satisfaction_col] >= 3]

    # Ensure there are enough eligible employees
    if len(eligible_employees) < num_members:
        st.error("Not enough eligible employees to fulfill the project requirement.")
        return False

    # Select employees for the project
    selected_employees = eligible_employees.sample(num_members)[emp_col].tolist()
    st.session_state.projects[project_name] = selected_employees
    st.success(f"Project '{project_name}' created with members: {', '.join(map(str, selected_employees))}")
    return True

# def add_project(project_name, num_members, df):
#     if 'EmpID' not in df.columns or 'JobSatisfaction' not in df.columns:
#         st.error("Please ensure that the dataset contains 'EmpID' and 'JobSatisfaction' columns.")
#         return False

#     if is_project_exists(project_name):
#         st.error(f"Project '{project_name}' already exists.")
#         return False

#     job_satisfaction_col = 'JobSatisfaction'
#     eligible_employees = df[df[job_satisfaction_col] >= 3]

#     if len(eligible_employees) < num_members:
#         st.error("Not enough eligible employees to fulfill the project requirement.")
#         return False

#     selected_employees = eligible_employees.sample(num_members).EmpID.tolist()
#     st.session_state.projects[project_name] = selected_employees
#     st.success(f"Project '{project_name}' created with members: {', '.join(map(str, selected_employees))}")
#     return True

def display_projects():
    if st.session_state.projects:
        st.write("### Created Projects")
        project_df = pd.DataFrame(
            [(name, ", ".join(map(str, members))) for name, members in st.session_state.projects.items()],
            columns=["Project Name", "Members (EmpID)"]
        )
        st.dataframe(project_df)

def landing_page():
    st.title("Welcome to WorkGEN")
    st.subheader("A Real-Time Data Insight Platform")
    st.write("**Workforce Analytics and People Management** helps you gain insights into your workforce data with interactive visualizations and automated analysis.")
    st.image("https://i.pinimg.com/originals/00/08/10/00081094ea8cf521ccebe03095ac0365.gif", caption="Analyze Your Workforce Effectively", use_column_width=True)
    st.markdown("### Key Features:\n- Upload CSV or Excel files with your workforce data.\n- Perform automated Exploratory Data Analysis (EDA).\n- Interactive visualizations like Bar Charts, Donut Charts, Bubble Charts, and Pie Charts.\n- Generate detailed and downloadable text-based insights on your dataset using Generative AI.")

def upload_and_preview():
    st.write("## Upload your CSV/Excel file")
    uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)

        df = rename_duplicate_columns(df)
        st.session_state.df = df

    if st.session_state.df is not None:
        st.write("### Dataset Preview")
        st.write(st.session_state.df.head())
        st.write("### Dataset Information")
        st.write(st.session_state.df.describe())

def generate_text_report(chart_type, df, x_axis=None, y_axis=None, size_col=None):
    if chart_type == "Bar Chart":
        insight = f"The Bar Chart visualizes the relationship between {x_axis} and {y_axis}. It appears that high values in {y_axis} are associated with {x_axis} variations. "
        insight += f"Some notable patterns are that the highest value in {y_axis} corresponds to {df[x_axis].iloc[df[y_axis].idxmax()]}. "
        insight += "This might indicate a trend worth further analysis."
        return insight

    elif chart_type == "Pie Chart":
        insight = f"The Pie Chart represents {x_axis} distribution. The largest segment is {df[x_axis].value_counts().idxmax()} with a count of {df[x_axis].value_counts().max()}, indicating this category's dominance. "
        insight += "This distribution might suggest preferences or population trends within this dataset."
        return insight

    elif chart_type == "Bubble Chart":
        insight = f"The Bubble Chart illustrates the interaction between {x_axis} and {y_axis} with bubble sizes based on {size_col}. "
        insight += f"Notably, the largest bubble is at {df[x_axis].iloc[df[size_col].idxmax()]} in {x_axis} and {df[y_axis].iloc[df[size_col].idxmax()]} in {y_axis}. "
        insight += "This correlation may reveal underlying factors impacting these values."
        return insight

    elif chart_type == "Donut Chart":
        insight = f"The Donut Chart shows {x_axis} proportions, with the largest section being {df[x_axis].value_counts().idxmax()}. "
        insight += "This visual helps to easily identify which categories take up the most share in the dataset."
        return insight

def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=3)
    return " ".join(str(sentence) for sentence in summary)


def visualization_and_text_gen():
    if st.session_state.df is None:
        st.error("Please upload a dataset first.")
        return
    
    st.write("### Data Visualizations")
    chart_type = st.selectbox("Select a chart type", ("Bar Chart", "Pie Chart", "Bubble Chart", "Donut Chart"))
    df = st.session_state.df
    
    if chart_type == "Bar Chart":
        x_axis = st.selectbox("Select X-axis", df.columns)
        y_axis = st.selectbox("Select Y-axis", df.columns)
        chart_id = (chart_type, x_axis, y_axis)

        if x_axis and y_axis and chart_id not in st.session_state.generated_charts:
            fig = px.bar(df, x=x_axis, y=y_axis, color=x_axis, color_continuous_scale=px.colors.sequential.Plasma)
            st.plotly_chart(fig)
            report = generate_text_report(chart_type, df, x_axis=x_axis, y_axis=y_axis)
            summarized_report = summarize_text(report)
            st.session_state.report_content.append(summarized_report)
            st.session_state.generated_charts.add(chart_id)

    elif chart_type == "Pie Chart":
        categorical_col = st.selectbox("Select Categorical Column", df.select_dtypes(include=['object']).columns)
        chart_id = (chart_type, categorical_col)
        if categorical_col and chart_id not in st.session_state.generated_charts:
            fig = px.pie(df, names=categorical_col)
            st.plotly_chart(fig)
            report = generate_text_report(chart_type, df, x_axis=categorical_col)
            summarized_report = summarize_text(report)
            st.session_state.report_content.append(summarized_report)
            st.session_state.generated_charts.add(chart_id)

    elif chart_type == "Bubble Chart":
        x_axis = st.selectbox("Select X-axis", df.select_dtypes(include='number').columns)
        y_axis = st.selectbox("Select Y-axis", df.select_dtypes(include='number').columns)
        size_col = st.selectbox("Select Size Column", df.select_dtypes(include='number').columns)
        chart_id = (chart_type, x_axis, y_axis, size_col)

        if x_axis and y_axis and size_col and chart_id not in st.session_state.generated_charts:
            fig = px.scatter(df, x=x_axis, y=y_axis, size=size_col, color=x_axis)
            st.plotly_chart(fig)
            report = generate_text_report(chart_type, df, x_axis=x_axis, y_axis=y_axis, size_col=size_col)
            summarized_report = summarize_text(report)
            st.session_state.report_content.append(summarized_report)
            st.session_state.generated_charts.add(chart_id)

    elif chart_type == "Donut Chart":
        categorical_col = st.selectbox("Select Categorical Column", df.select_dtypes(include=['object']).columns)
        chart_id = (chart_type, categorical_col)
        
        if categorical_col and chart_id not in st.session_state.generated_charts:
            fig = px.pie(df, names=categorical_col, hole=0.5)
            st.plotly_chart(fig)
            report = generate_text_report(chart_type, df, x_axis=categorical_col)
            summarized_report = summarize_text(report)
            st.session_state.report_content.append(summarized_report)
            st.session_state.generated_charts.add(chart_id)

    if st.session_state.report_content:
        st.write("### Generated Text Report")
        for report in st.session_state.report_content:
            st.write(report)

    if st.session_state.report_content:
        st.download_button(
            label="Download Report as Text File",
            data="\n\n".join(st.session_state.report_content),
            file_name="analysis_report.txt",
            mime="text/plain"
        )

    if st.session_state.report_content:
        st.download_button(
            label="Download Report as Word File",
            data="\n\n".join(st.session_state.report_content),
            file_name="analysis_report.doc",
            mime="text/plain"
        )

def autoviz_page(df):
    if df is None:
        st.error("Please upload a dataset first.")
        return

    if st.checkbox("Run AutoViz for Automated EDA"):
        st.write("Auto-generated Visualizations")
        AV = AutoViz_Class()
        df_av = AV.AutoViz(
            filename="",
            sep=",",
            depVar="",
            dfte=df,
            header=0,
            verbose=0,
            lowess=False,
            chart_format="png",
            max_rows_analyzed=150000,
            max_cols_analyzed=30,
        )
        for fig in plt.get_fignums():
            st.pyplot(plt.figure(fig))
        
        st.session_state.autoviz_run = True

def project_creation_page():
    st.write("## Create New Project")
    project_name = st.text_input("Project Name")
    num_members = st.number_input("Number of Members", min_value=1, step=1)

    if st.button("Create Project"):
        if st.session_state.df is None:
            st.error("Please upload a dataset first.")
        elif not project_name:
            st.error("Project Name cannot be empty.")
        else:
            add_project(project_name, int(num_members), st.session_state.df)
    
    display_projects()

page_options = {
    "Landing": landing_page,
    "Upload and Data Preview": upload_and_preview,
    "Visualization and Text Generation": visualization_and_text_gen,
    "AutoViz Visualization": lambda: autoviz_page(st.session_state.df) if st.session_state.df is not None else st.error("Please upload a dataset first."),
    "Project Creation": project_creation_page,
}


selected_page = st.sidebar.selectbox("Select a Page", list(page_options.keys()))
page_options[selected_page]()
