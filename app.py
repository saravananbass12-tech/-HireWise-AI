import os
import io
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ==================================================
# HIREWISE AI - JOB PREDICTION SYSTEM
# ==================================================

st.set_page_config(
    page_title="HireWise AI",
    page_icon="💼",
    layout="wide"
)

st.title("💼 HireWise AI")
st.subheader("Human Job Search and Recruitment Outcome Prediction")
st.write(
    "Predict job search responses, interview results, and application "
    "status using Random Forest Classification."
)
st.divider()


# ==================================================
# 1. DATA SOURCE: DEFAULT DATASET, DOWNLOAD, UPLOAD
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(
    BASE_DIR,
    "human_job_prediction_50000_records_converted.csv"
)

FEATURE_COLUMNS = [
    "Education",
    "Experience_Years",
    "Skills_Score",
    "Applications_Sent",
    "Coding_Score",
    "Communication_Score",
    "Interview_Attended",
]

TARGET_COLUMNS = [
    "Job_Search_Status",
    "Interview_Result",
    "Application_Status",
]

REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS

TARGET_NAMES = {
    "Job_Search_Status": "Job Search Status",
    "Interview_Result": "Interview Result",
    "Application_Status": "Application Status",
}

CATEGORICAL_FEATURES = ["Education"]
NUMERIC_FEATURES = [
    "Experience_Years",
    "Skills_Score",
    "Applications_Sent",
    "Coding_Score",
    "Communication_Score",
    "Interview_Attended",
]

st.sidebar.header("📁 Data Source")

template_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
template_csv = template_df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="📥 Download Dataset Format (CSV)",
    data=template_csv,
    file_name="hirewise_ai_dataset_template.csv",
    mime="text/csv",
    use_container_width=True,
)

st.sidebar.divider()
st.sidebar.subheader("📤 Upload your custom CSV Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV containing the required input and target columns.",
)
st.sidebar.caption("Maximum file size: 200 MB")


@st.cache_data(show_spinner=False)
def load_data(source_bytes):
    """Read a CSV from bytes and standardize common column names."""
    data = pd.read_csv(io.BytesIO(source_bytes))
    data.columns = data.columns.astype(str).str.strip()

    normalized = {
        str(column).strip().lower().replace(" ", "_").replace("-", "_"): column
        for column in data.columns
    }

    aliases = {
        "Education": ["education", "education_level", "qualification"],
        "Experience_Years": [
            "experience_years", "experience", "years_experience",
            "work_experience"
        ],
        "Skills_Score": [
            "skills_score", "skill_score", "skills_rating", "skill_rating"
        ],
        "Applications_Sent": [
            "applications_sent", "applications", "total_applications",
            "jobs_applied"
        ],
        "Coding_Score": [
            "coding_score", "coding_test_score", "technical_score",
            "technical_test_score"
        ],
        "Communication_Score": [
            "communication_score", "communication_test_score",
            "communication_rating"
        ],
        "Interview_Attended": [
            "interview_attended", "interviews_attended", "interview_count",
            "number_of_interviews"
        ],
        "Job_Search_Status": [
            "job_search_status", "job_response_status", "response_status",
            "job_search_outcome"
        ],
        "Interview_Result": [
            "interview_result", "interview_status", "interview_outcome",
            "interview_decision"
        ],
        "Application_Status": [
            "application_status", "job_selection_status", "selection_status",
            "selection_result", "job_status", "hiring_status", "final_status",
            "job_outcome", "recruitment_status", "application_result"
        ],
    }

    rename_map = {}
    for standard_name, possible_names in aliases.items():
        if standard_name in data.columns:
            continue
        for possible_name in possible_names:
            actual_name = normalized.get(possible_name)
            if actual_name is not None:
                rename_map[actual_name] = standard_name
                break

    if rename_map:
        data = data.rename(columns=rename_map)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
            + ". Download the CSV template from the sidebar and use its "
            "column names."
        )

    # Normalize application outcome labels.
    data["Application_Status"] = (
        data["Application_Status"].astype("string").str.strip().replace({
            "Selected": "Approved",
            "selected": "Approved",
            "Not Selected": "Rejected",
            "Not selected": "Rejected",
            "Rejected Application": "Rejected",
        })
    )

    # Clean target labels.
    for column in TARGET_COLUMNS:
        data[column] = data[column].astype("string").str.strip()
        data[column] = data[column].replace(
            ["", "nan", "None", "<NA>"], pd.NA
        )

    # Convert numerical input columns.
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["Education"] = data["Education"].astype("string").str.strip()
    data["Education"] = data["Education"].replace(
        ["", "nan", "None", "<NA>"], pd.NA
    )

    return data


# Choose the uploaded CSV if available; otherwise use the default file.
try:
    if uploaded_file is not None:
        df = load_data(uploaded_file.getvalue())
        st.sidebar.success("✅ Custom CSV uploaded successfully!")

        st.sidebar.download_button(
            label="📄 Download Loaded Dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="hirewise_ai_loaded_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        if not os.path.isfile(DATA_FILE):
            st.error(
                "Dataset file not found. Upload a CSV using the sidebar or "
                "place 'human_job_prediction_50000_records_converted.csv' "
                "in the same folder as app.py."
            )
            st.stop()

        with open(DATA_FILE, "rb") as dataset_file:
            df = load_data(dataset_file.read())

except Exception as error:
    st.error(f"❌ Dataset loading error: {error}")
    st.stop()

if df.empty:
    st.error("The dataset is empty. Please upload a CSV with data rows.")
    st.stop()

st.success(f"✅ Dataset loaded successfully: {len(df):,} records")


# ==================================================
# 2. TRAIN RANDOM FOREST MODELS
# ==================================================

@st.cache_resource(show_spinner=False)
def train_models(dataset_bytes):
    """Train one Random Forest pipeline for each target column."""
    training_data = load_data(dataset_bytes)
    models = {}
    metrics = {}

    for target in TARGET_COLUMNS:
        model_data = training_data[FEATURE_COLUMNS + [target]].dropna(
            subset=[target]
        ).copy()

        model_data = model_data[
            model_data[target].astype("string").str.strip().ne("")
        ]

        if model_data.empty:
            raise ValueError(f"No valid target values were found for {target}.")

        if model_data[target].nunique() < 2:
            raise ValueError(
                f"{target} must contain at least two different classes."
            )

        class_counts = model_data[target].value_counts()
        if class_counts.min() < 2:
            raise ValueError(
                f"Every class in {target} needs at least two records "
                "for stratified splitting. Add more examples for each class."
            )

        X = model_data[FEATURE_COLUMNS]
        y = model_data[target].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "education",
                    Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    CATEGORICAL_FEATURES,
                ),
                (
                    "numbers",
                    SimpleImputer(strategy="median"),
                    NUMERIC_FEATURES,
                ),
            ]
        )

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        models[target] = model
        metrics[target] = {
            "accuracy": accuracy,
            "y_test": y_test,
            "predictions": predictions,
            "classes": model.named_steps["classifier"].classes_,
        }

    return models, metrics


# Keep the exact loaded CSV bytes as the training cache key.
try:
    if uploaded_file is not None:
        active_dataset_bytes = uploaded_file.getvalue()
    else:
        with open(DATA_FILE, "rb") as dataset_file:
            active_dataset_bytes = dataset_file.read()

    with st.spinner("Training Random Forest models... Please wait."):
        models, model_metrics = train_models(active_dataset_bytes)

except Exception as error:
    st.error(f"❌ Model Training Error: {error}")
    st.stop()


# ==================================================
# 3. MODEL ACCURACY DASHBOARD
# ==================================================

st.header("📊 Model Accuracy & Performance Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric(
    "Job Search Accuracy",
    f"{model_metrics['Job_Search_Status']['accuracy'] * 100:.2f}%"
)
col3.metric(
    "Interview Accuracy",
    f"{model_metrics['Interview_Result']['accuracy'] * 100:.2f}%"
)
col4.metric(
    "Application Accuracy",
    f"{model_metrics['Application_Status']['accuracy'] * 100:.2f}%"
)

st.markdown("<br>", unsafe_allow_html=True)


# ==================================================
# 4. CONFUSION MATRIX
# ==================================================

st.header("🎯 Confusion Matrix")

selected_task = st.selectbox(
    "Select Prediction Task",
    options=TARGET_COLUMNS,
    format_func=lambda target: TARGET_NAMES[target],
)

selected_metrics = model_metrics[selected_task]
y_test = selected_metrics["y_test"]
predictions = selected_metrics["predictions"]
classes = selected_metrics["classes"]

cm = confusion_matrix(y_test, predictions, labels=classes)

fig_cm = px.imshow(
    cm,
    x=classes,
    y=classes,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Blues",
    labels={
        "x": "Predicted Result",
        "y": "Actual Result",
        "color": "Count",
    },
    title=f"Confusion Matrix - {TARGET_NAMES[selected_task]}",
)
st.plotly_chart(fig_cm, use_container_width=True)


# ==================================================
# 5. CANDIDATE PREDICTION FORM
# ==================================================

st.header("🤖 Candidate Prediction Form")
st.write("Enter candidate details to predict recruitment outcomes.")

education_options = sorted(
    df["Education"].dropna().astype(str).unique().tolist()
)

if not education_options:
    st.error("No valid Education values were found in the dataset.")
    st.stop()

with st.form("candidate_form"):
    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox("Education", options=education_options)
        experience = st.number_input(
            "Experience (Years)", min_value=0, max_value=50, value=1, step=1
        )
        skills_score = st.slider(
            "Skills Score", min_value=0, max_value=100, value=70
        )
        applications_sent = st.number_input(
            "Applications Sent", min_value=0, max_value=1000, value=10
        )

    with col2:
        coding_score = st.slider(
            "Coding Score", min_value=0, max_value=100, value=70
        )
        communication_score = st.slider(
            "Communication Score", min_value=0, max_value=100, value=70
        )
        interview_attended = st.number_input(
            "Interviews Attended", min_value=0, max_value=100, value=1
        )

    submit = st.form_submit_button(
        "🔮 Predict Outcomes", use_container_width=True
    )


# ==================================================
# 6. PREDICTION RESULTS
# ==================================================

if submit:
    candidate = pd.DataFrame([{
        "Education": education,
        "Experience_Years": experience,
        "Skills_Score": skills_score,
        "Applications_Sent": applications_sent,
        "Coding_Score": coding_score,
        "Communication_Score": communication_score,
        "Interview_Attended": interview_attended,
    }])

    st.divider()
    st.header("✅ Prediction Results")
    result_columns = st.columns(3)

    for index, target in enumerate(TARGET_COLUMNS):
        model = models[target]
        prediction = model.predict(candidate)[0]
        probabilities = model.predict_proba(candidate)[0]
        confidence = float(probabilities.max() * 100)

        with result_columns[index]:
            st.subheader(TARGET_NAMES[target])
            st.success(f"Predicted Result: {prediction}")
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

    st.subheader("Candidate Input Details")
    st.dataframe(candidate, use_container_width=True)
    st.caption(
        "Confidence represents the model's highest predicted class "
        "probability. It is not a guarantee of actual recruitment outcomes."
    )


# ==================================================
# 7. DATASET PREVIEW
# ==================================================

st.divider()

with st.expander("📁 View Dataset"):
    st.dataframe(df.head(20), use_container_width=True)


# ==================================================
# 8. PROJECT INFORMATION
# ==================================================

with st.expander("ℹ️ About HireWise AI"):
    st.write("""
    **Project Name:** HireWise AI

    **Algorithm:** Random Forest Classification

    **Programming Language:** Python

    **Libraries:** Streamlit, Pandas, Scikit-learn, Plotly

    **Input Features:**
    - Education
    - Experience Years
    - Skills Score
    - Applications Sent
    - Coding Score
    - Communication Score
    - Interviews Attended

    **Output Predictions:**
    - Job Search Status
    - Interview Result
    - Application Status
    

    **Dataset:** Human job prediction dataset.

    **Author : SARAVANAN.D**
    """)

st.divider()
st.caption("HireWise AI | Random Forest Classification | Python | Streamlit")