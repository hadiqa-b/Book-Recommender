# 📚 Book Recommender with Sentence Transformers & marimo

This project is a simple, content-based book recommendation system built using:

- 🧠 [Sentence Transformers](https://www.sbert.net/) for semantic understanding of book descriptions
- ⚡ [polars](https://pola.rs/) for fast and efficient data processing
- 🧩 [marimo](https://marimo.io/) to explore interactive Python notebooks with reactive cells and UI

---

## 🎯 Project Goal

The primary goal of this project was to **test out marimo**, a modern alternative to Jupyter notebooks. I used a Kaggle book dataset to build a small semantic search system that:

- Accepts a **book description** as input
- Uses **Sentence Transformers** to embed the input and compare it to a dataset of book descriptions
- Returns the **top N most similar books** based on cosine similarity

This project was kept intentionally simple and clean to focus on experimenting with marimo's workflow.

---

## 📁 Dataset

The dataset used is:  
**Books Dataset for NLP and Recommendation Systems** by `sinatavakoli` on Kaggle.

We use [kagglehub](https://pypi.org/project/kagglehub/) to automatically download the dataset.

The dataset includes:
- Book titles
- Descriptions
- (Optionally) authors, genres, etc.

---

## 📥 Dataset Download

You can run the data_loader.py file to get the dataset.


## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| `marimo` | Interactive notebook engine |
| `sentence-transformers` | Embedding natural language book descriptions |
| `torch` | Backend for transformer models |
| `polars` | DataFrame library for speed and clarity |
| `kagglehub` | Lightweight tool to fetch datasets from Kaggle|

---

## 🚀 How to Run

1. Clone this repo or copy the main notebook file.
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Launch the marimo app (if using .py notebook):

```bash
marimo run book_recommender.py
```
4. You can edit the notebook as well:
```bash
marimo edit
```
5. Enter any book description in the UI and get instant recommendations!