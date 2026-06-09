from har_utils.model import model_random_forest, model_hist_gradient
from har_utils.visualization import (plot_confusion_matrix)
from sklearn.metrics import classification_report
from har_utils.data import get_X_y_split
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import numpy as np
from har_utils.preprocessing import preprocess_all_subjects
import matplotlib.pyplot as plt

def compare_feature_description(original_df, transformed_df, feature=None):
    print("Original")
    print(original_df[feature].describe())
    
    print("\nNormalized")
    print(transformed_df[feature].describe())

def evaluate_random_forest_base(X_train, y_train, X_test, y_test):
    rf = model_random_forest()
    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    print(
        classification_report(
            y_test,
            pred,
            zero_division=0
        )
    )
    print('\n\n')
    plot_confusion_matrix(y_test, pred)

    return pred

def evaluate_random_forest(all_subject, random_state=42):
    X_train, y_train, X_test, y_test = get_X_y_split(all_subject, random_state)
    evaluate_random_forest_base(X_train, y_train, X_test, y_test)
    

def evaluate_loso(all_subjects, model):
    """
    Leave-One-Subject-Out cross-validation.

    Parameters
    ----------
    all_subjects : dict
        Keys are subject IDs (strings), values are windowed DataFrames
        with features and 'Action Label' column.

    Returns
    -------
    dict
        Per-subject macro F1 scores.
    """
    subject_ids  = list(all_subjects.keys())
    label_col    = 'Action Label'
    fold_results = {}

    for test_subject in subject_ids:
        train_subjects = [s for s in subject_ids if s != test_subject]

        train_df = pd.concat(
            [all_subjects[s] for s in train_subjects],
            ignore_index=True
        )
        test_df = all_subjects[test_subject]

        X_train = train_df.drop(columns=[label_col])
        y_train = train_df[label_col]
        X_test  = test_df.drop(columns=[label_col])
        y_test  = test_df[label_col]

        if model is None:
            print('No Model provided! Using Default: Random Forest')
            model = model_random_forest()
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        macro_f1 = f1_score(y_test, preds, average='macro')
        report   = classification_report(y_test, preds, output_dict=True, zero_division=0)
        fold_results[test_subject] = {
            'macro_f1': macro_f1,
            'report':   report,
            'y_test':   y_test,
            'preds':    preds
        }

        # ── Per-fold summary ──────────────────────────────────────
        print(f"\n{'─' * 52}")
        print(f"  Fold — Test Subject: {test_subject}  |  Train: {train_subjects}")
        print(f"{'─' * 52}")

        # Per-class F1 as a compact table
        classes = sorted(y_test.unique())
        print(f"  {'Class':<22} {'Precision':>9} {'Recall':>9} {'F1':>9}")
        print(f"  {'─'*22} {'─'*9} {'─'*9} {'─'*9}")
        for cls in classes:
            r = report.get(cls, {})
            p  = r.get('precision', 0)
            rc = r.get('recall',    0)
            f1 = r.get('f1-score',  0)
            print(f"  {cls:<22} {p:>9.2f} {rc:>9.2f} {f1:>9.2f}")

        print(f"\n  Macro F1: {macro_f1:.3f}")

    # ── LOSO summary ─────────────────────────────────────────────
    scores = [v['macro_f1'] for v in fold_results.values()]
    mean   = np.mean(scores)
    std    = np.std(scores)

    print(f"\n{'═' * 52}")
    print(f"  LOSO RESULTS SUMMARY")
    print(f"{'═' * 52}")
    print(f"  {'Subject':<12} {'Macro F1':>10}")
    print(f"  {'─'*12} {'─'*10}")
    for subj, res in fold_results.items():
        print(f"  {subj:<12} {res['macro_f1']:>10.3f}")
    print(f"  {'─'*12} {'─'*10}")
    print(f"  {'Mean':<12} {mean:>10.3f}")
    print(f"  {'Std':<12} {std:>10.3f}")
    print(f"{'═' * 52}\n")

    return fold_results

def preprocess_evaluate_random_forest(all_subject, window_size):
    all_subject_processed = preprocess_all_subjects(all_subject, window_size)
    return evaluate_loso(all_subject_processed)

def get_feature_importance(all_subjects, label_col='Action Label', top_n=30):
    """
    Trains one Random Forest on all subjects combined and extracts
    feature importances. This gives a global view of which features
    the model finds most useful across the entire dataset.

    Parameters
    ----------
    all_subjects : dict
        Keys are subject IDs, values are windowed DataFrames.
    label_col : str
        Name of the label column.
    top_n : int
        How many top features to display.

    Returns
    -------
    pd.DataFrame
        Feature importances sorted descending.
    """
    combined = pd.concat(all_subjects.values(), ignore_index=True)

    X = combined.drop(columns=[label_col])
    y = combined[label_col]

    model = model_random_forest()
    model.fit(X, y)

    importance_df = pd.DataFrame({
        'feature':    X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # ── Print top N ───────────────────────────────────────────────
    print(f"\n{'═' * 52}")
    print(f"  TOP {top_n} FEATURE IMPORTANCES")
    print(f"{'═' * 52}")
    print(f"  {'Rank':<6} {'Feature':<40} {'Importance':>10}")
    print(f"  {'─'*6} {'─'*40} {'─'*10}")
    for i, row in importance_df.head(top_n).iterrows():
        print(f"  {i+1:<6} {row['feature']:<40} {row['importance']:>10.4f}")
    print(f"{'═' * 52}\n")

    # ── Bar chart ─────────────────────────────────────────────────
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, top_n * 0.35))
    bars = ax.barh(
        top['feature'][::-1],
        top['importance'][::-1]
    )
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

    return importance_df