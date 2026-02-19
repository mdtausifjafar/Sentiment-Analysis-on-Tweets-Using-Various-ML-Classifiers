# Sentiment Analysis on Tweets Using Various ML Classifiers

A machine learning project that classifies tweets into **Positive**, **Negative**, or **Neutral** sentiments using multiple classical ML classifiers trained on TF-IDF features, achieving **~88% accuracy**.

---

## Overview

Social media platforms like Twitter generate massive amounts of opinionated text every day. Understanding the sentiment behind these tweets is valuable for brand monitoring, market research, and public opinion analysis.

This project builds a complete **Tweet Sentiment Analysis pipeline** that:

- Cleans and preprocesses raw tweet text
- Extracts rich features using multiple TF-IDF vectorizers
- Trains and compares three high-performance ML classifiers
- Automatically selects the best-performing model
- Provides a ready-to-use prediction function for new tweets

---

## Dataset

| Property                | Detail                                                   |
| ----------------------- | -------------------------------------------------------- |
| **File**          | `Dataset/twitter_dataset.csv`                          |
| **Total Samples** | ~38,000 tweets                                           |
| **Classes**       | Positive, Negative, Neutral, Irrelevant                              |
| **Split**         | 80% Training / 20% Test (stratified)                     |

### Class Mapping

The original dataset contains 4 labels. The `Irrelevant` class is merged into `Neutral` according to the project instruction:

| Original Label | Mapped To |
| -------------- | --------- |
| Positive       | Positive  |
| Negative       | Negative  |
| Neutral        | Neutral   |
| Irrelevant     | Neutral   |

### Columns Used

- `tweet` — raw tweet text (input)
- `tweet_sentiment` — sentiment label (target)

Columns `tweet_id` and `tweet_concept` are dropped as they are not needed for modeling.

---

## Preprocessing Pipeline

Each tweet goes through the following steps before being fed into the model:

1. **Lowercase conversion** — all text normalized to lowercase
2. **Contraction expansion** — `n't` → `not`, `won't` → `will not`, `can't` → `cannot`
3. **URL & username removal** — `http://...`, `@user` stripped
4. **Hashtag cleaning** — `#` symbol removed, word kept
5. **Character deduplication** — `loooove` → `love` (max 2 repeats)
6. **Sentiment markers** — `!` → adds token `exclamation`; `?` → adds token `question`
7. **Negation compound tokens** — `not bad` → `not_bad` (fixes bag-of-words negation problem)
8. **Stopword removal** — using NLTK, keeping sentiment-preserving words (`not`, `never`, `very`, `really`, etc.)
9. **Lemmatization** — words reduced to base form using WordNetLemmatizer

---

## Feature Engineering

Three TF-IDF vectorizers are combined into a single feature matrix:

| Vectorizer     | Type            | Features | N-gram Range | Purpose                                       |
| -------------- | --------------- | -------- | ------------ | --------------------------------------------- |
| Word TF-IDF    | Word-level      | 8,000    | (1, 2)       | Captures words and two-word phrases           |
| Char TF-IDF    | Character-level | 4,000    | (2, 5)       | Handles slang, abbreviations, spelling errors |
| Trigram TF-IDF | Word-level      | 3,000    | (3, 3)       | Captures three-word context                   |

> **Total feature dimensions: 15,000**

All vectorizers use `sublinear_tf=True` (log normalization), `min_df=2`, and `max_df=0.90` to remove very rare and very common terms.

The three feature matrices are horizontally stacked using `scipy.sparse.hstack`.

---

## Models

Three classifiers are trained and evaluated:

### 1. Logistic Regression

- Fast and reliable baseline for text classification
- Handles high-dimensional sparse TF-IDF data well
- **Key settings**: `C=10.0`, `solver='saga'`, `class_weight='balanced'`, `max_iter=2000`

### 2. Linear SVM (`LinearSVC`)

- Excellent for high-dimensional, linearly separable data
- Significantly faster than kernel SVM
- **Key settings**: `C=2.0`, `class_weight='balanced'`, `max_iter=3000`

### 3. SGD Classifier

- Stochastic gradient descent — very fast on large datasets
- Uses `modified_huber` loss (similar to SVM, but supports probabilities)
- **Key settings**: `loss='modified_huber'`, `alpha=1e-5`, `class_weight='balanced'`

---

## Results

All models are ranked by accuracy on the **20% held-out test set**:

| Rank | Model                | Accuracy         |
| ---- | -------------------- | ---------------- |
| 1    | **Linear SVM** | **88.20%** |
| 2    | SGD Classifier       | 87.93%           |
| 3    | Logistic Regression  | 87.57%           |

### Classification Report — Linear SVM (Best Model)

| Class             | Precision      | Recall         | F1-Score       |
| ----------------- | -------------- | -------------- | -------------- |
| Negative          | 0.89           | 0.88           | 0.88           |
| Neutral           | 0.88           | 0.90           | 0.89           |
| Positive          | 0.89           | 0.85           | 0.87           |
| **Overall** | **0.88** | **0.88** | **0.88** |

> The model automatically selects and uses the **highest-accuracy model** for predictions. No model names are hardcoded — the selection is fully dynamic.

---

## Libraries Used

| Library          | Purpose                               |
| ---------------- | ------------------------------------- |
| `pandas`       | Data loading and manipulation         |
| `numpy`        | Numerical operations                  |
| `re`           | Regular expressions for text cleaning |
| `nltk`         | Stopword list, lemmatization          |
| `scikit-learn` | ML models, TF-IDF, evaluation metrics |
| `scipy`        | Sparse matrix operations (`hstack`) |
| `matplotlib`   | Visualization                         |
| `seaborn`      | Heatmap for confusion matrix          |

### Install Dependencies

```bash
pip install pandas numpy nltk scikit-learn scipy matplotlib seaborn
```

---

## Project Structure

```
Sentiment Analysis on Tweets Using Various ML Classifiers/
│
├── Dataset/
│   └── twitter_dataset.csv
│
├── Sentiment_Analysis_on_Tweets_Using_Various_ML_Classifiers.ipynb
│
└── README.md
```

---

## How to Run

1. Clone or download the repository
2. Place the dataset at `Dataset/twitter_dataset.csv`
3. Open the notebook in **Jupyter Notebook** or **Google Colab**
4. Run all cells top-to-bottom

---

## Making Predictions

After running the notebook, use the `predict_sentiment()` function to classify any tweet:

```python
tweet = "I absolutely love this product! Best purchase ever!"
sentiment, probabilities = predict_sentiment(tweet)

print(sentiment)       # → "Positive"
print(probabilities)   # → {'Positive': 1.000, 'Negative': 0.000, 'Neutral': 0.000}
```

The function automatically uses the **best-performing model** found during training.
