# Resume-Job Description Matcher (NLP)

## Business problem
Job seekers apply with the same resume to many different roles, and
recruiters/ATS systems screen resumes for keyword and semantic
relevance to a specific job description. This project builds a tool
that scores how well a resume matches a target JD, flags missing
skills, and classifies the resume into a job category — turning
tailoring guesswork into something measurable.

## Data
[Kaggle: Resume Dataset by snehaanbhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
— 2,484 labeled resumes across job categories (HR, IT, Teacher, Finance,
Sales, Designer, etc.), used to train the category classifier.

## Approach

### Part A — Resume category classifier (supervised NLP)
1. **Text cleaning**: stripped URLs, emails, punctuation, and numbers —
   resume text copy-pasted from PDFs is noisy and these add no
   classification signal.
2. **TF-IDF vectorization** with unigrams + bigrams (capped at 5,000
   features) — bigrams catch multi-word skills like "machine learning"
   that unigrams break apart and lose meaning.
3. **Compared 3 models**: Multinomial Naive Bayes, Logistic Regression,
   Linear SVM — stratified train/test split to preserve category
   balance.
4. Picked the best by weighted F1 (accounts for class imbalance across
   the ~25 categories, some with far fewer resumes than others).

### Part B — Resume-to-JD semantic matcher (the practical tool)
1. **Sentence embeddings** (`all-MiniLM-L6-v2`) turn the full resume
   and JD text into vectors capturing *meaning*, not just exact words —
   catches cases like "led a team" matching "managed people" that
   pure keyword/TF-IDF matching would miss.
2. **Cosine similarity** between the two embeddings gives a 0-100%
   match score.
3. **Skill-gap analysis**: a curated skill keyword list (not a trained
   NER model — deliberately simple and explainable) flags which
   JD-required skills are present vs. missing in the resume.
4. **Deployed as a Streamlit app**: paste resume + JD text, get match
   score, skill gaps, and predicted category in one view.

## Results
*(fill in your actual numbers after running `eda_and_modeling.py`)*

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Multinomial Naive Bayes | — | — |
| Logistic Regression | — | — |
| Linear SVM | — | — |

## What I'd improve with more time
- Replace the curated skill list with a trained NER model for skill
  extraction, once enough labeled data is available
- Fine-tune the sentence embedding model on resume/JD pairs specifically,
  rather than using a general-purpose pretrained model
- Add resume section parsing (separate Experience/Education/Skills)
  for more granular matching instead of treating the resume as one blob

## How to run
```bash
pip install -r requirements.txt
mkdir -p data outputs
# place Resume.csv (from the snehaanbhawal Kaggle dataset) inside data/
python eda_and_modeling.py
streamlit run app.py
```

## Live demo
(add your deployed Streamlit Community Cloud link here)
