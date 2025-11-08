"""
toxicity_model.py
This module implements a simple toxicity classifier using scikit‑learn.  The
classifier is a logistic regression model trained on bag‑of‑words features.

Functions
---------
read_data(path: str) -> pandas.DataFrame
    Read a TSV file into a DataFrame. Assumes the first column is named
    ``text`` and the second column is ``label`` or ``demographic``.

train_classifier(train_df: pd.DataFrame) -> (Pipeline, TfidfVectorizer)
    Build a text classification pipeline using TF‑IDF features and
    logistic regression.  Returns the fitted pipeline.

evaluate_model(clf: Pipeline, dev_df: pd.DataFrame) -> dict
    Evaluate a classifier on a development set.  Returns a dictionary of
    accuracy, precision, recall and F1 scores for each class.

compute_fpr(preds: np.ndarray, groups: Sequence[str]) -> Dict[str, float]
    Compute the false positive rate for each demographic group.  Assumes
    all examples are non‑offensive (i.e., true label ``NOT``), so any
    prediction of ``OFF`` counts as a false positive.

predict_and_save(clf: Pipeline, test_df: pd.DataFrame, outfile: str) -> None
    Use a classifier to predict labels on the test set and save them to
    ``outfile``.  The output file will contain a header ``label`` and one
    prediction per line.

This code is designed to be reusable and modular.  You can import the
functions in a notebook or run them from the command line to perform the
assignment tasks.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def read_data(path: str) -> pd.DataFrame:
    """Read a TSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the TSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.
    """
    return pd.read_csv(path, sep="\t")


def train_classifier(train_df: pd.DataFrame) -> Pipeline:
    """Train a logistic regression classifier on text data.

    This function builds a pipeline with a TF‑IDF vectorizer and a
    logistic regression classifier.  It uses a limited set of features
    (unigrams and bigrams) to capture surface‑level information in the
    tweets.  The logistic regression is configured with a relatively high
    number of maximum iterations to ensure convergence.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with columns ``text`` and ``label``.

    Returns
    -------
    Pipeline
        Fitted scikit‑learn pipeline.
    """
    # A simple bag‑of‑words representation.  We found through grid search
    # that unigrams with a larger feature space (~10k features) provided
    # the best macro F1 on the development set while still meeting the
    # efficiency requirements for this assignment.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 1),
        max_features=10000,
    )
    # Use class_weight='balanced' to mitigate class imbalance between OFF
    # and NOT labels.  Increase max_iter to ensure convergence.
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])
    pipeline.fit(train_df["text"], train_df["label"])
    return pipeline


def evaluate_model(clf: Pipeline, dev_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate a classifier on a development set.

    Computes accuracy as well as precision, recall and F1 scores for
    each class.  Returns a dictionary mapping metric names to values.

    Parameters
    ----------
    clf : Pipeline
        Fitted text classification pipeline.
    dev_df : pd.DataFrame
        Development data with columns ``text`` and ``label``.

    Returns
    -------
    dict
        A dictionary containing accuracy, precision, recall, and F1 for
        each class in alphabetical order (NOT, OFF).
    """
    y_true = dev_df["label"]
    y_pred = clf.predict(dev_df["text"])
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["NOT", "OFF"], average=None
    )
    return {
        "accuracy": accuracy,
        "precision_NOT": precision[0],
        "recall_NOT": recall[0],
        "f1_NOT": f1[0],
        "precision_OFF": precision[1],
        "recall_OFF": recall[1],
        "f1_OFF": f1[1],
    }


def compute_fpr(preds: Sequence[str], groups: Sequence[str]) -> Dict[str, float]:
    """Compute the false positive rate (FPR) per demographic group.

    Assumes that none of the examples are truly offensive; therefore
    any prediction of ``OFF`` is a false positive.  The FPR is the
    fraction of examples in each group that are falsely predicted as
    offensive.

    Parameters
    ----------
    preds : sequence of str
        Predicted labels (each either ``OFF`` or ``NOT``).
    groups : sequence of str
        Demographic group labels corresponding to each prediction.

    Returns
    -------
    dict
        Mapping from group name to false positive rate.
    """
    fpr: Dict[str, float] = {}
    unique_groups = sorted(set(groups))
    preds = np.array(list(preds))
    groups = np.array(list(groups))
    for grp in unique_groups:
        mask = groups == grp
        if mask.sum() == 0:
            fpr[grp] = 0.0
            continue
        fp = np.sum(preds[mask] == "OFF")
        fpr[grp] = fp / mask.sum()
    return fpr


def predict_and_save(clf: Pipeline, test_df: pd.DataFrame, outfile: str) -> None:
    """Predict labels for the test set and save to a TSV file.

    Parameters
    ----------
    clf : Pipeline
        Fitted classifier.
    test_df : pd.DataFrame
        Test data with a ``text`` column.
    outfile : str
        Path to output TSV file.  The file will contain a single
        column named ``label`` with predicted labels.
    """
    preds = clf.predict(test_df["text"])
    pd.DataFrame({"label": preds}).to_csv(outfile, sep="\t", index=False)


def rule_based_perspective(dev_df: pd.DataFrame, threshold: float = 0.8) -> np.ndarray:
    """Classify tweets based on PerspectiveAPI scores.

    A simple rule‑based classifier that labels a tweet as offensive
    (``OFF``) if its ``perspective_score`` exceeds the given threshold.
    Otherwise it labels it as non‑offensive (``NOT``).

    Parameters
    ----------
    dev_df : pd.DataFrame
        Development data containing columns ``perspective_score``.
    threshold : float, optional
        Toxicity threshold above which a tweet is considered offensive.
        Defaults to 0.8.

    Returns
    -------
    np.ndarray
        Array of predicted labels (``OFF`` or ``NOT``).
    """
    scores = dev_df["perspective_score"].astype(float)
    return np.where(scores > threshold, "OFF", "NOT")


def main():
    """Example entry point for running training and evaluation.

    This function demonstrates how to use the above utilities to train a
    classifier, evaluate it, and save predictions.  It is not run
    automatically when the module is imported.  To execute, run
    ``python toxicity_model.py`` from the command line.  The function
    assumes the data files are located in a ``civility_data``
    directory relative to the working directory.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate toxicity classifier")
    parser.add_argument("--data_dir", default="civility_data", help="Path to data directory")
    parser.add_argument("--output", default="FirstName_LastName_test.tsv", help="Output predictions file")
    args = parser.parse_args()

    train_df = read_data(f"{args.data_dir}/train.tsv")
    dev_df = read_data(f"{args.data_dir}/dev.tsv")
    test_df = read_data(f"{args.data_dir}/test.tsv")

    clf = train_classifier(train_df)
    results = evaluate_model(clf, dev_df)
    print("Evaluation results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # Compute FPR on mini_demographic_dev.tsv
    demo_df = read_data(f"{args.data_dir}/mini_demographic_dev.tsv")
    demo_preds = clf.predict(demo_df["text"])
    fpr = compute_fpr(demo_preds, demo_df["demographic"])
    print("\nFalse positive rate per demographic group:")
    for grp, rate in fpr.items():
        print(f"{grp}: {rate:.4f}")

    # Save predictions for test set
    predict_and_save(clf, test_df, args.output)


if __name__ == "__main__":
    main()