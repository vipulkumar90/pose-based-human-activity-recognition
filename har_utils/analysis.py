from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from har_utils.model import model_random_forest
from har_utils.visualization import plot_confusion_matrix
from sklearn.metrics import classification_report, f1_score
from har_utils.data import get_X_y_split, get_X_y_split_single_subject, print_clean_header
import pandas as pd
import numpy as np
from har_utils.preprocessing import preprocess_all_subjects
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import f1_score
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from har_utils.model import build_lstm_model
from har_utils.preprocessing import scale_sequences


# English:
# Purpose: Compare feature distributions before and after transformation by printing descriptive statistics.
# Parameters:
# - original_df: DataFrame with original feature values.
# - transformed_df: DataFrame with transformed feature values.
# - feature: Name of the feature column to compare.
# Returns:
# - None. Side effect: Prints summary statistics to console.
# 日本語:
# 目的: 特徴量の要約統計量を表示して、変換前後の分布を比較します。
# パラメータ:
# - original_df: 元の特徴量値を含むDataFrame。
# - transformed_df: 変換後の特徴量値を含むDataFrame。
# - feature: 比較する特徴量列の名前。
# 戻り値:
# - なし。副作用: 要約統計量をコンソールに出力します。
def compare_feature_description(original_df, transformed_df, feature=None):
    print("Original")
    # Show summary statistics for the original feature values.
    # 元の特徴量の要約統計量を表示します。
    print(original_df[feature].describe())

    print("\nNormalized")
    # Show summary statistics for the transformed feature values.
    # 変換後の特徴量の要約統計量を表示します。
    print(transformed_df[feature].describe())

# English:
#
# 日本語:
#
def get_loso_folds(all_subjects):
    subject_ids = list(all_subjects.keys())
    label_col = 'Action Label'
    loso_folds = []
    for test_subject in subject_ids:
        # Build a list of subjects to use for training in this fold.
        # このフォールドで学習に使用する被験者リストを作成します。
        train_subjects = [s for s in subject_ids if s != test_subject]

        # Concatenate data for all training subjects.
        # すべての学習対象被験者のデータを連結します。
        train_df = pd.concat(
            [all_subjects[s] for s in train_subjects],
            ignore_index=True
        )
        test_df = all_subjects[test_subject]

        # Separate features and labels for training and testing.
        # 学習用とテスト用の特徴量とラベルを分離します。
        X_train = train_df.drop(columns=[label_col])
        y_train = train_df[label_col]
        X_test = test_df.drop(columns=[label_col])
        y_test = test_df[label_col]
        loso_folds.append([X_train, y_train, X_test, y_test, test_subject, train_subjects])

    return loso_folds

# English:
# Purpose: Perform leave-one-subject-out cross-validation for model evaluation.
# For each subject, train on all other subjects and evaluate on the held-out subject.
# Parameters:
# - all_subjects: Dictionary mapping subject IDs to their DataFrames.
# - model: Classifier model with fit() and predict() methods. If None, uses default Random Forest.
# - verbose: Determines how much information needs to be printed. ['None', 'Medium', 'All']
# Returns:
# - Dictionary with per-subject results including macro F1 scores, detailed reports, and predictions.
# Side effect: Prints per-fold summaries and overall LOSO results summary to console.
# 日本語:
# 目的: モデル評価のためにleave-one-subject-outクロスバリデーションを実行します。
# 各被験者を除外し、残りの被験者でモデルを学習して評価します。
# パラメータ:
# - all_subjects: 被験者IDをデータフレームにマッピングする辞書。
# - model: fit()およびpredict()メソッドを持つ分類器モデル。Noneの場合はデフォルトランダムフォレストを使用。
# 戻り値:
# - マクロF1スコア、詳細レポート、予測を含む被験者別の結果の辞書。
# 副作用: 各フォールドのサマリーと全体的なLOSO結果をコンソールに出力します。
def evaluate_loso(all_subjects, model, verbose='None'):
    # Gather all subject identifiers from the dictionary.
    # 辞書からすべての被験者IDを取得します。
    fold_results = {}

    loso_folds = get_loso_folds(all_subjects)

    for X_train, y_train, X_test, y_test, test_subject, train_subjects in loso_folds:
        if model is None:
            print('No Model provided! Using Default: Random Forest')
            # Use default model if none is supplied.
            # モデルが提供されていない場合は、デフォルトのモデルを使用します。
            model = model_random_forest()
        if verbose == 'Medium' or verbose == 'All':
            print(f"Model being evaluated: {model.__class__.__name__}")

        # Encode the labels in case of XGBoost
        y_train, y_test = prepare_labels(model, y_train, y_test)
        
        # Train the model on the current fold's training data.
        # 現在のフォールドの学習データでモデルを学習させます。
        model.fit(X_train, y_train)

        # Predict labels on the hold-out subject.
        # 保留されたテスト被験者に対してラベルを予測します。
        preds = model.predict(X_test)

        # Compute macro F1 and detailed report for this fold.
        # このフォールドのマクロF1と詳細レポートを計算します。
        macro_f1 = f1_score(y_test, preds, average='macro')
        report = classification_report(
            y_test,
            preds,
            output_dict=True,
            zero_division=0
        )

        fold_results[test_subject] = {
            'macro_f1': macro_f1,
            'report': report,
            'y_test': y_test,
            'preds': preds
        }

        if verbose == 'All':
            # Print a summary for the current fold.
            # 現在のフォールドのサマリーを表示します。
            print_clean_header(f"\nFold — Test Subject: {test_subject}  |  Train: {train_subjects}",
                            border_char='-',
                            line_len=52)

            classes = sorted(y_test.unique())
            print(f"  {'Class':<22} {'Precision':>9} {'Recall':>9} {'F1':>9}")
            print(f"  {'─'*22} {'─'*9} {'─'*9} {'─'*9}")
            for cls in classes:
                # Use safe fallback values if the class is missing from report.
                # レポートにクラスが含まれていない場合は、デフォルト値を使用します。
                r = report.get(cls, {})
                p = r.get('precision', 0)
                rc = r.get('recall', 0)
                f1 = r.get('f1-score', 0)
                print(f"  {cls:<22} {p:>9.2f} {rc:>9.2f} {f1:>9.2f}")

            # Print macro F1 score for the fold.
            # フォールドのマクロF1スコアを表示します。
            print(f"\n  Macro F1: {macro_f1:.3f}")

    # Compute summary statistics across all folds.
    # すべてのフォールドにわたる集計統計を計算します。
    scores = [v['macro_f1'] for v in fold_results.values()]
    mean = np.mean(scores)
    std = np.std(scores)

    if verbose == 'Medium' or verbose == 'All':
        print_clean_header(f"LOSO RESULTS SUMMARY", line_len=52)
        print(f"  {'Subject':<12} {'Macro F1':>10}")
        print(f"  {'─'*12} {'─'*10}")
        for subj, res in fold_results.items():
            print(f"  {subj:<12} {res['macro_f1']:>10.3f}")
        print(f"  {'─'*12} {'─'*10}")
        print(f"  {'Mean':<12} {mean:>10.3f}")
        print(f"  {'Std':<12} {std:>10.3f}")
        print(f"{'═' * 52}\n")

    return fold_results



# English:
# Purpose: Apply preprocessing with windowing and then evaluate using LOSO cross-validation.
# Parameters:
# - all_subject: Dictionary mapping subject IDs to their DataFrames.
# - window_size: Size of the sliding window for preprocessing in frames.
# Returns:
# - Dictionary with per-subject LOSO cross-validation results from evaluate_loso.
# 日本語:
# 目的: ウィンドウサイズを指定して前処理を行い、LOSOクロスバリデーションで評価します。
# パラメータ:
# - all_subject: 被験者IDをデータフレームにマッピングする辞書。
# - window_size: 前処理用のスライディングウィンドウサイズ（フレーム数）。
# 戻り値:
# - evaluate_losoからの被験者別LOSOクロスバリデーション結果の辞書。
def preprocess_evaluate_loso_supervised(all_subject, model, window_size, stride, verbose=False, drop_bad_shoulder_frames=False):
    # Apply preprocessing to every subject with the requested window size.
    # 指定されたウィンドウサイズで各被験者の前処理を適用します。
    all_subject_processed = preprocess_all_subjects(all_subject, window_size=window_size, stride=stride, drop_bad_shoulder_frames=drop_bad_shoulder_frames)

    # Evaluate the preprocessed data using LOSO.
    # 前処理済みデータをLOSOで評価します。
    return evaluate_loso(all_subject_processed, model, verbose)



# English:
# Purpose: Train a Random Forest on all subjects and extract/visualize top feature importances.
# Parameters:
# - all_subjects: Dictionary mapping subject IDs to their DataFrames.
# - label_col: Name of the label column (default: 'Action Label').
# - top_n: Number of top features to display and plot (default: 30).
# Returns:
# - DataFrame with feature names and their importance scores, sorted descending.
# Side effect: Prints ranked feature importance table and displays horizontal bar chart.
# 日本語:
# 目的: すべての被験者でランダムフォレストを学習し、上位特徴量重要度を抽出・可視化します。
# パラメータ:
# - all_subjects: 被験者IDをデータフレームにマッピングする辞書。
# - label_col: ラベル列の名前（デフォルト: 'Action Label'）。
# - top_n: 表示およびプロットする上位特徴量の数（デフォルト: 30）。
# 戻り値:
# - 特徴量名と重要度スコアを含むDataFrame（降順でソート）。
# 副作用: ランク付けされた特徴量重要度テーブルを表示し、横棒グラフを表示します。
def get_feature_importance(all_subjects, label_col='Action Label', top_n=30, model=None, verbose=False):
    # Combine the data of all subjects into a single DataFrame.
    # すべての被験者のデータを1つのDataFrameに結合します。
    combined = pd.concat(all_subjects.values(), ignore_index=True)

    # Separate features and labels from the combined dataset.
    # 結合データセットから特徴量とラベルを分離します。
    X = combined.drop(columns=[label_col])
    y = combined[label_col]

    # Train a Random Forest on the full dataset.
    # 全データセットでランダムフォレストを学習させます。
    if model == None:
        model = model_random_forest() # Default Model

    if verbose:
        print(f"Model being evaluated: {model.__class__.__name__}")
    if model.__class__.__name__ == "XGBClassifier":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    model.fit(X, y)

    # Extract feature importances and sort them descending.
    # 特徴量重要度を抽出し、降順でソートします。
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # Print the top features in the console.
    # 上位の特徴量をコンソールに表示します。
    print(f"\n{'═' * 52}")
    print(f"  TOP {top_n} FEATURE IMPORTANCES")
    print(f"{'═' * 52}")
    print(f"  {'Rank':<6} {'Feature':<40} {'Importance':>10}")
    print(f"  {'─'*6} {'─'*40} {'─'*10}")
    for i, row in importance_df.head(top_n).iterrows():
        print(f"  {i+1:<6} {row['feature']:<40} {row['importance']:>10.4f}")
    print(f"{'═' * 52}\n")

    # Plot the top feature importances as a horizontal bar chart.
    # 上位の特徴量重要度を横棒グラフとして表示します。
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, top_n * 0.35))
    ax.barh(
        top['feature'][::-1],
        top['importance'][::-1]
    )
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

    return importance_df

def predict_supervised(X_test, y_test, model):

    # Predict labels for the test set.
    # テストデータに対して予測を行う。
    preds = model.predict(X_test)

    # Print classification report and plot confusion matrix.
    # 分類レポートを表示し、混同行列をプロットする。
    print_clean_header('Classification Report', border_char='*')
    print(classification_report(y_test, preds, zero_division=0))

    # 4. Plot confusion matrix to visualize prediction errors.
    # 4. 予測エラーを可視化するために混同行列をプロットする。
    print_clean_header('Confusion Matrix', border_char='*')
    plot_confusion_matrix(y_test, preds)

def evaluate_supervised(X_train, y_train, X_test, y_test, model, verbose=False):

    if verbose:
        # Print the model being evaluated.
        # 評価対象のモデル名を表示する。
        print_clean_header(f"Evaluating Model: {model.__class__.__name__}")

    # Train the provided model on the training data.
    # モデルを学習する。
    model.fit(X_train, y_train)
    
    preds = predict_supervised(X_test, y_test, model)

    # Return the predicted labels for potential further analysis.
    # さらなる分析のために予測ラベルを返す。
    return preds

# English:
# Purpose:
# Filter each subject dataset by retaining only the top N most important features
# based on the provided feature importance ranking.
#
# Parameters:
# - all_subjects: Dictionary where keys are subject IDs and values are DataFrames.
# - importance_df: DataFrame containing feature importance rankings. Must include
#                  a 'feature' column sorted in descending order of importance.
# - top_n: Number of most important features to retain (default: 150).
# - label_col: Name of the target label column, which is always retained
#              (default: "Action Label").
#
# Returns:
# - Dictionary containing the filtered subject DataFrames with the selected
#   features and the label column.
#
# 日本語:
# 目的:
# 特徴量重要度のランキングに基づいて、各被験者データセットから
# 上位N個の重要な特徴量のみを保持します。
#
# パラメータ:
# - all_subjects: キーが被験者ID、値がDataFrameである辞書。
# - importance_df: 特徴量重要度ランキングを含むDataFrame。
#                  重要度の高い順に並んだ 'feature' 列を含む必要があります。
# - top_n: 保持する重要特徴量の数（デフォルト: 150）。
# - label_col: 常に保持する目的ラベル列の名前
#              （デフォルト: "Action Label"）。
#
# 戻り値:
# - 選択された特徴量とラベル列のみを含む被験者DataFrameの辞書。
def filter_features_by_importance(all_subjects, importance_df, top_n=150, label_col='Action Label', verbose=False):
    # Select the top N most important features.
    # 上位N個の重要特徴量を取得する。
    top_features = importance_df['feature'].head(top_n).tolist()

    # Always include the label column.
    # ラベル列は常に保持する。
    cols_to_keep = top_features + [label_col]

    filtered = {}

    # Filter each subject dataset using the selected features.
    # 各被験者データセットを選択した特徴量でフィルタリングする。
    for subject_id, df in all_subjects.items():
        # Only keep columns that actually exist in this df
        # (guards against any mismatch between importance_df and subject columns)
        # 現在のDataFrameに存在する列のみを保持する。
        #（存在しない特徴量によるエラーを防ぐ。）
        available = [c for c in cols_to_keep if c in df.columns]
        filtered[subject_id] = df[available].copy()

    # Display a summary of the filtering results.
    # フィルタリング結果の概要を表示する。
    print(f"Features kept: {top_n}")
    print(f"Features dropped: {len(importance_df) - top_n}")
    # print(f"Subjects processed: {list(filtered.keys())}")

    return filtered

def evaluate_lstm_loso(
    loso_sequences,
    shared_encoder,
    n_epochs=20,
    batch_size=32,
    lstm_units=64,
    sequence_length=90
):
    """
    Leave-One-Subject-Out evaluation for LSTM.
    Mirrors evaluate_random_forest_loso structure exactly
    so results are directly comparable.

    Parameters
    ----------
    loso_sequences : dict
        Output of build_loso_sequence_datasets.
        Keys are subject IDs, values are dicts with 'X' and 'y'.
    shared_encoder : LabelEncoder
        Single encoder fitted across all subjects.
    n_epochs : int
        Maximum training epochs. Early stopping may stop before this.
    batch_size : int
        Training batch size.
    lstm_units : int
        LSTM hidden units (paper used 64).
    sequence_length : int
        Frames per sequence — must match what was used in
        build_loso_sequence_datasets.
    """
    subject_ids = list(loso_sequences.keys())
    n_classes   = len(shared_encoder.classes_)
    n_features  = loso_sequences[subject_ids[0]]['X'].shape[2]
    fold_results = {}

    for test_subject in subject_ids:
        train_subjects = [s for s in subject_ids if s != test_subject]

        print(f"\n{'─' * 52}")
        print(f"  Fold — Test Subject: {test_subject}  "
              f"|  Train: {train_subjects}")
        print(f"{'─' * 52}")

        # ── Assemble train and test arrays ────────────────────────
        X_train = np.concatenate(
            [loso_sequences[s]['X'] for s in train_subjects], axis=0
        )
        y_train = np.concatenate(
            [loso_sequences[s]['y'] for s in train_subjects], axis=0
        )
        X_test = loso_sequences[test_subject]['X']
        y_test = loso_sequences[test_subject]['y']

        # ── Scale features ────────────────────────────────────────
        X_train, X_test = scale_sequences(X_train, X_test)

        # ── One-hot encode labels for Keras ───────────────────────
        # LSTM with softmax needs one-hot targets, not integers
        y_train_ohe = categorical(y_train, num_classes=n_classes)

        # ── Build fresh model for each fold ───────────────────────
        # Important: must rebuild to avoid weights carrying over
        tf.keras.backend.clear_session()
        model = build_lstm_model(sequence_length, n_features, n_classes, lstm_units)

        # ── Early stopping — stops if val_loss stops improving ────
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=0
        )

        # ── Train ─────────────────────────────────────────────────
        history = model.fit(
            X_train, y_train_ohe,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0          # suppress per-epoch output
        )

        epochs_run = len(history.history['loss'])
        print(f"  Trained for {epochs_run} epochs")

        # ── Predict ───────────────────────────────────────────────
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred       = np.argmax(y_pred_proba, axis=1)

        # ── Evaluate ──────────────────────────────────────────────
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        report   = classification_report(
            y_test, y_pred,
            target_names=shared_encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        fold_results[test_subject] = {
            'macro_f1': macro_f1,
            'report':   report,
            'y_test':   y_test,
            'preds':    y_pred,
            'history':  history.history
        }

        # ── Per-fold summary ──────────────────────────────────────
        classes = shared_encoder.classes_
        print(f"\n  {'Class':<22} {'Precision':>9} {'Recall':>9} {'F1':>9}")
        print(f"  {'─'*22} {'─'*9} {'─'*9} {'─'*9}")
        for cls in classes:
            r  = report.get(cls, {})
            p  = r.get('precision', 0)
            rc = r.get('recall',    0)
            f1 = r.get('f1-score',  0)
            print(f"  {cls:<22} {p:>9.2f} {rc:>9.2f} {f1:>9.2f}")

        print(f"\n  Macro F1: {macro_f1:.3f}")

    # ── LOSO summary ──────────────────────────────────────────────
    scores = [v['macro_f1'] for v in fold_results.values()]
    mean   = np.mean(scores)
    std    = np.std(scores)

    print(f"\n{'═' * 52}")
    print(f"  LSTM LOSO RESULTS SUMMARY")
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

def prepare_labels(model, y_train, y_test):
    """
    Encode labels only for models that require numeric targets.
    """
    if isinstance(model, XGBClassifier):
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

    return y_train, y_test