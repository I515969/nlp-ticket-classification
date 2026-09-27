# Automatic Customer Support Ticket Classification

**IIITB Post Graduate Programme in AI & ML — NLP Case Study**
**Author:** Pritam Saha (I515969)

---

## Problem Statement

A financial company receives a large volume of unstructured customer support tickets across products such as credit cards, bank accounts, and loans. Manually routing each ticket to the right department is tedious and does not scale.

The goal of this case study is to build an NLP-based model that:
1. Automatically identifies the **topic/category** of each complaint using unsupervised **topic modelling (NMF)**
2. Trains a **supervised classifier** on the discovered labels to route any new incoming complaint to the correct department

The five target categories are:
- Credit card / Prepaid card
- Bank account services
- Theft / Dispute reporting
- Mortgages / Loans
- Others

---

## Approach & Pipeline

| Step | Description |
|---|---|
| 1. Data Loading | Load raw complaints from JSON, normalise into a dataframe |
| 2. Data Cleaning | Rename columns, replace blanks with NaN, remove null complaints |
| 3. Text Preprocessing | Lowercase, remove punctuation/numbers/brackets, lemmatize, POS filter (nouns only) |
| 4. EDA | Character length distribution, word cloud (top 40), unigram/bigram/trigram frequency |
| 5. Feature Extraction | TF-IDF vectorization (`max_df=0.95`, `min_df=2`) → Document Term Matrix |
| 6. Topic Modelling | NMF with 5 components to discover latent complaint categories |
| 7. Model Building | Train Logistic Regression, Decision Tree, Random Forest on NMF-derived labels |
| 8. Model Inference | Use best model to predict category for new unseen complaints |

---

## Dataset

| Property | Value |
|---|---|
| File | `complaints-2021-05-14_08_16.json` |
| Source | US Consumer Financial Protection Bureau (CFPB) |
| Total records | 78,313 |
| Records after cleaning | 21,072 |
| Key column | `complaint_what_happened` |

---

## Technology & Environment

| Component | Detail |
|---|---|
| Language | Python 3.14 |
| Environment | Virtual environment (`nlp_venv`) |
| Notebook | Jupyter (executed in Cursor IDE) |

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `numpy` | 2.5.3 | Numerical operations |
| `pandas` | 3.0.6 | Data manipulation |
| `re` / `string` | stdlib | Text cleaning |
| `nltk` | 3.10.3 | NLP utilities |
| `spacy` + `en_core_web_sm` | 3.8.16 | Lemmatization & POS tagging |
| `matplotlib` | 3.11.2 | Visualisation |
| `seaborn` | 0.13.2 | Visualisation |
| `plotly` | 7.1.0 | Interactive charts |
| `wordcloud` | 1.9.6 | Word cloud generation |
| `scikit-learn` | 1.9.1 | TF-IDF, NMF, ML models, evaluation |

---

## Results

### Topic Modelling (NMF)

| Topic | Top Words | Assigned Label |
|---|---|---|
| 0 | account, bank, deposit, checking, transfer | Bank Account services |
| 1 | card, charge, dispute, fraud, purchase | Credit card or prepaid card |
| 2 | loan, mortgage, home, foreclosure, property | Mortgage/Loan |
| 3 | inquiry, report, identity, theft, bureau | Theft/Dispute Reporting |
| 4 | payment, fee, statement, auto, error | Others |

### Supervised Model Comparison

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Logistic Regression** | **92.41%** | **0.92** |
| Random Forest | 80.85% | 0.77 |
| Decision Tree | 79.53% | 0.79 |

**Best Model: Logistic Regression** — selected for inference.

---

## Key Findings

- NMF with 5 components produced well-separated, interpretable topic clusters that aligned directly with the business-defined categories.
- Logistic Regression outperformed tree-based models by a significant margin (~12%), achieving 92.4% accuracy with balanced precision and recall across all five categories.
- Theft/Dispute Reporting and Others showed slightly lower recall, reflecting overlap in complaint language across categories.
- The trained model can be deployed to automatically route incoming complaints, significantly reducing manual effort and resolution time.

---

## Repository Structure

```
.
├── I515969_Pritam_Saha_Automatic_Ticket_Classification_Assignment.ipynb
├── complaints-2021-05-14_08_16.json
└── README.md
```
