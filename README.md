# Advanced NLP: Sentiment Analysis & Clinical Text Structuring

Two applied NLP pipelines built with modern and classical approaches: sentiment classification comparing a traditional ML model against a pretrained Transformer, and a clinical NLP pipeline that extracts and structures medical information from unstructured clinical reports.

## Project 1 — Sentiment Analysis: Naive Bayes vs BERT/RoBERTa

Compares a classical machine learning approach against a pretrained Transformer model on the same sentiment classification task (positive / negative / neutral).

- **Naive Bayes**: Multinomial Naive Bayes with TF-IDF (unigrams + bigrams). Accuracy ~0.65, strong on positive/negative, weaker on neutral (frequently confused with the other classes).
- **BERT/RoBERTa**: pretrained multilingual sentiment model, used zero-shot (no fine-tuning on this dataset). Accuracy ~0.64 — comparable to Naive Bayes, but with better recall on the neutral class at the cost of precision, showing a tendency to default to "neutral" under ambiguity.
- **Takeaway**: a classical TF-IDF-based model can match a pretrained Transformer's accuracy on this task, highlighting that model choice should depend on the specific class balance and use case, not just architecture sophistication.

## Project 2 — Structuring Unstructured Clinical Reports

Simulates a real hospital use case: clinical reports are typically stored as unstructured free text, making systematic, quantitative analysis difficult. This pipeline extracts and structures the clinically relevant information automatically.

- **Preprocessing**: light NLTK-based cleaning tailored to clinical text — numbers, symbols, and negations are preserved (they carry clinical meaning), while NER is run on the *original*, uncleaned text to avoid losing information.
- **Named Entity Recognition**: uses **SciSpaCy**'s biomedical model (`en_core_sci_sm`) to identify clinically relevant mentions (symptoms, diagnoses, treatments) directly from real clinical transcription reports.
- **Structuring**: extracted entities are reshaped into a tabular format (one row per entity, linked to its source report and medical specialty), turning unstructured clinical text into analyzable data — the kind of transformation needed to feed hospital systems or clinical databases.
- **Pattern analysis by specialty**: entity frequencies are broken down by hospital specialty (e.g., Cardiovascular/Pulmonary, Bariatrics, Chiropractic), surfacing domain-specific recurring terms (e.g., "catheter", "proximal" in Cardiovascular; "abdomen", "weight loss" in Bariatrics).
- **Qualitative evaluation**: a manual inspection comparing original text vs. extracted entities across several reports, to assess real-world applicability and identify limitations — the model's generalist tagging scheme labels most entities simply as `ENTITY` rather than distinguishing disease/symptom/treatment, which would require a more specialized clinical NER model in a production hospital setting.

## Tech stack

- Python
- NLTK, spaCy, **SciSpaCy** (`en_core_sci_sm`)
- scikit-learn (TF-IDF, Multinomial Naive Bayes)
- Hugging Face Transformers (pretrained multilingual sentiment model)
- pandas, matplotlib

## Data

- Project 1: social media / user review text (sentiment-labeled).
- Project 2: real clinical transcription reports (medical specialty-labeled), sourced from a public Kaggle dataset.

## Notes

This project began as a group activity, where each member completed their own individual implementation before the group met to compare approaches and refine conclusions together. The code, analysis, and interpretation in this notebook are my own individual work. The original assignment brief is not included here as it belongs to the issuing institution.
