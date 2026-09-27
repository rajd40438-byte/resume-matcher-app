"""
A curated list of common tech/business skills used for keyword-based
skill extraction from resume and job description text. This is a
simple, transparent, and explainable approach — an alternative to a
black-box NER model, and easy to explain in an interview: "I used a
curated skill taxonomy rather than a trained NER model because it's
more precise for a well-defined domain and doesn't need labeled
training data."

Feel free to extend this list for your own target roles.
"""

SKILLS = [
    # Programming languages
    "python", "r", "sql", "java", "c++", "javascript", "scala", "matlab",
    # Data science / ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data analysis", "data visualization", "statistics",
    "regression", "classification", "clustering", "time series",
    "feature engineering", "hyperparameter tuning", "model evaluation",
    "xgboost", "lightgbm", "catboost", "random forest", "logistic regression",
    "neural networks", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn",
    # Data engineering / tools
    "excel", "power bi", "tableau", "spark", "hadoop", "airflow", "etl",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
    "streamlit", "flask", "django", "rest api", "fastapi",
    # Business / soft skills often screened for
    "project management", "communication", "leadership", "stakeholder management",
    "agile", "scrum", "presentation", "problem solving",
    # Analytics-specific
    "a/b testing", "customer segmentation", "forecasting", "rfm",
    "customer lifetime value", "churn analysis", "smote",
]
