# Amazon Reviews Analysis

A Jupyter notebook project exploring Amazon product reviews through natural language processing, time-series analysis, and neural-network classification. The analysis uses a small dataset of 997 reviews across 40 products, so findings and model scores are exploratory rather than broadly generalisable.

## Project files

- `Summartive.ipynb` — the analysis notebook.
- `amazon_reviews_lab.csv` — input dataset required by the notebook.

Place both files in the same directory before running the notebook. The notebook reads the CSV using the relative path `amazon_reviews_lab.csv`.

## What the notebook does

1. **Review text analysis:** inspects ratings and review lengths; cleans, tokenizes, and lemmatizes review text; compares word and phrase frequencies; and contrasts low- and high-rated reviews with smoothed log-odds.
2. **Time-series analysis:** aggregates reviews by month, examines trends and stationarity, explores seasonal patterns, compares review-volume forecasting models, and produces an illustrative six-month forecast.
3. **Rating classification:** converts review text to TF-IDF features and trains a multilayer perceptron (MLP) to predict 1–5 star ratings. It evaluates class-level metrics and baselines, explores activation/optimizer choices, and includes a PyTorch implementation for comparison.

The notebook uses a fixed random seed (`42`) for reproducibility where supported. Neural-network results can still vary across library versions, hardware, and backends.

## Dataset

The CSV contains 997 review records and these columns:

| Column | Description |
| --- | --- |
| `product_id` | Product identifier |
| `rating` | Star rating from 1 to 5 |
| `review_text` | Full review text used for text analysis and classification |
| `review_summary` | Short review headline |
| `review_time_raw` | Original review date text |
| `unix_review_time` | Unix timestamp representation |
| `timestamp` | Parsed review date used for monthly analysis |
| `sentiment_label` | Sentiment category derived from rating; excluded from rating prediction to avoid target leakage |
| `review_length_words` | Review length supplied in the dataset |

The rating classes are imbalanced, with a majority of 5-star reviews and relatively few low-star reviews. The notebook discusses this limitation and reports metrics beyond accuracy.

## Setup

Use Python 3.10 or newer and Jupyter Notebook/Lab. Install the core packages:

```bash
python -m pip install jupyter pandas numpy matplotlib scikit-learn statsmodels nltk tensorflow
```

The notebook can fall back to scikit-learn stop words and a lightweight rule-based lemmatizer if NLTK or its language resources are unavailable. TensorFlow/Keras is used when available for the neural-network experiments; the notebook includes a scikit-learn MLP fallback. PyTorch is an optional dependency for the framework comparison in Part 3:

```bash
python -m pip install torch
```

If TensorFlow is unavailable, the notebook may use its scikit-learn fallback for the core classification work, but the activation/optimizer experiment specifically requires Keras. The time-series section requires `statsmodels`.

## Run the notebook

1. Put `Summartive.ipynb` and `amazon_reviews_lab.csv` in one folder.
2. Open a terminal in that folder and launch Jupyter:

   ```bash
   jupyter lab
   ```

3. Open `Summartive.ipynb` and run the cells from top to bottom. The notebook generates charts and printed summaries inline.

The notebook may attempt a one-time NLTK resource download when those resources are not installed. All other dependencies should be installed before running it.

## Interpretation notes

- The dataset has only 997 reviews, and ratings are strongly skewed toward positive reviews.
- Early dates are sparse; monthly modelling uses the notebook's selected consecutive review window.
- The six-month forecast is illustrative and should not be treated as a production forecast.
- Classification performance for minority ratings is uncertain because each class has few examples and evaluation uses a single held-out split.
- Product mix, text-cleaning decisions, package versions, and neural-network randomness can affect the results.
