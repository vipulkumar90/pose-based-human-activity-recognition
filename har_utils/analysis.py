from har_utils.model import model_random_forest
from har_utils.visualization import plot_confusion_matrix
from sklearn.metrics import classification_report, f1_score
from har_utils.data import get_X_y_split
import pandas as pd
import numpy as np
from har_utils.preprocessing import preprocess_all_subjects
import matplotlib.pyplot as plt



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
# Purpose: Train a Random Forest classifier and evaluate it on a test split.
# Parameters:
# - X_train: Training feature matrix.
# - y_train: Training labels.
# - X_test: Test feature matrix.
# - y_test: Test labels.
# Returns:
# - Predicted labels for the test set as a numpy array.
# Side effect: Prints classification report and plots confusion matrix to console.
# 日本語:
# 目的: ランダムフォレスト分類器をトレーニングしテストデータで評価します。
# パラメータ:
# - X_train: 学習用特徴量行列。
# - y_train: 学習用ラベル。
# - X_test: テスト用特徴量行列。
# - y_test: テスト用ラベル。
# 戻り値:
# - テストセットの予測ラベル（numpy配列）。
# 副作用: 分類レポートを表示し、混同行列をコンソールにプロットします。
def evaluate_random_forest_base(X_train, y_train, X_test, y_test):
    # Create the default Random Forest model.
    # デフォルトのランダムフォレストモデルを作成します。
    rf = model_random_forest()

    # Train the Random Forest on the provided training data.
    # 提供された学習データでランダムフォレストを学習させます。
    rf.fit(X_train, y_train)

    # Predict labels for the test set using the trained model.
    # 学習済みモデルを使ってテストセットのラベルを予測します。
    pred = rf.predict(X_test)

    # Print the classification report with precision, recall, and F1.
    # 精度、再現率、F1を含む分類レポートを表示します。
    print(
        classification_report(
            y_test,
            pred,
            zero_division=0
        )
    )
    print('\n\n')

    # Plot confusion matrix to visualize prediction errors.
    # 予測誤りを可視化するために混同行列をプロットします。
    plot_confusion_matrix(y_test, pred)

    return pred



# English:
# Purpose: Run Random Forest evaluation using a train/test split with a random test subject.
# Parameters:
# - all_subject: Complete dataset containing all samples and labels from all subjects.
# - random_state: Seed for reproducible random splits (default: 42).
# Returns:
# - None. Side effect: Prints classification report and confusion matrix to console via evaluate_random_forest_base.
# 日本語:
# 目的: ランダム選択されたテスト被験者を使用して、train/test分割でランダムフォレスト評価を実行します。
# パラメータ:
# - all_subject: すべての被験者からのすべてのサンプルとラベルを含む完全なデータセット。
# - random_state: 再現可能なランダム分割のシード（デフォルト: 42）。
# 戻り値:
# - なし。副作用: evaluate_random_forest_baseを経由してコンソールに分類レポートと混同行列を出力します。
def evaluate_random_forest(all_subject, random_state=42):
    # Split the full dataset into training and testing sets.
    # 全データセットを学習用とテスト用に分割します。
    X_train, y_train, X_test, y_test = get_X_y_split(all_subject, random_state)

    # Evaluate the Random Forest on the split data.
    # 分割データでランダムフォレストを評価します。
    evaluate_random_forest_base(X_train, y_train, X_test, y_test)



# English:
# Purpose: Perform leave-one-subject-out cross-validation for model evaluation.
# For each subject, train on all other subjects and evaluate on the held-out subject.
# Parameters:
# - all_subjects: Dictionary mapping subject IDs to their DataFrames.
# - model: Classifier model with fit() and predict() methods. If None, uses default Random Forest.
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
def evaluate_loso(all_subjects, model):
    # Gather all subject identifiers from the dictionary.
    # 辞書からすべての被験者IDを取得します。
    subject_ids = list(all_subjects.keys())
    label_col = 'Action Label'
    fold_results = {}

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

        if model is None:
            print('No Model provided! Using Default: Random Forest')
            # Use default model if none is supplied.
            # モデルが提供されていない場合は、デフォルトのモデルを使用します。
            model = model_random_forest()

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

        # Print a summary for the current fold.
        # 現在のフォールドのサマリーを表示します。
        print(f"\n{'─' * 52}")
        print(f"  Fold — Test Subject: {test_subject}  |  Train: {train_subjects}")
        print(f"{'─' * 52}")

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
def preprocess_evaluate_random_forest(all_subject, window_size):
    # Apply preprocessing to every subject with the requested window size.
    # 指定されたウィンドウサイズで各被験者の前処理を適用します。
    all_subject_processed = preprocess_all_subjects(all_subject, window_size)

    # Evaluate the preprocessed data using LOSO.
    # 前処理済みデータをLOSOで評価します。
    return evaluate_loso(all_subject_processed)



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
def get_feature_importance(all_subjects, label_col='Action Label', top_n=30):
    # Combine the data of all subjects into a single DataFrame.
    # すべての被験者のデータを1つのDataFrameに結合します。
    combined = pd.concat(all_subjects.values(), ignore_index=True)

    # Separate features and labels from the combined dataset.
    # 結合データセットから特徴量とラベルを分離します。
    X = combined.drop(columns=[label_col])
    y = combined[label_col]

    # Train a Random Forest on the full dataset.
    # 全データセットでランダムフォレストを学習させます。
    model = model_random_forest()
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