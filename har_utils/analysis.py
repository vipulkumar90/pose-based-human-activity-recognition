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

        print(f"Model being evaluated: {model.__class__.__name__}")

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
def preprocess_evaluate_loso_supervised(all_subject, model, window_size, stride, verbose=False):
    # Apply preprocessing to every subject with the requested window size.
    # 指定されたウィンドウサイズで各被験者の前処理を適用します。
    all_subject_processed = preprocess_all_subjects(all_subject, window_size, stride)

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

def evaluate_supervised(all_subjects, model):

    # 0. Print the model being evaluated.
    # 0. 評価対象のモデル名を表示する。
    print_clean_header(f"Evaluating Model: {model.__class__.__name__}")
    
    # 1. Get train/test split for the provided subjects.
    # 1. 学習用データとテスト用データを取得する。
    if len(all_subjects) < 1:
        raise ValueError("At least 1 subject is required for evaluation.")
    elif len(all_subjects) < 2:
        X_train, y_train, X_test, y_test = get_X_y_split_single_subject(list(subjects.values())[0])
    else:
        X_train, y_train, X_test, y_test = get_X_y_split(subjects)

    # 2. Train the provided model on the training data.
    # 2. モデルを学習する。
    model.fit(X_train, y_train)

    # 3. Predict labels for the test set.
    # 3. テストデータに対して予測を行う。
    preds = model.predict(X_test)

    # 4. Print classification report and plot confusion matrix.
    # 4. 分類レポートを表示し、混同行列をプロットする。
    print_clean_header('Classification Report', border_char='*')
    print(classification_report(y_test, preds, zero_division=0))

    # 5. Plot confusion matrix to visualize prediction errors.
    # 5. 予測エラーを可視化するために混同行列をプロットする。
    print_clean_header('Confusion Matrix', border_char='*')
    plot_confusion_matrix(y_test, preds)

    # 6. Return the predicted labels for potential further analysis.
    # 6. さらなる分析のために予測ラベルを返す。
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
def filter_features_by_importance(all_subjects, importance_df, top_n=150, label_col='Action Label'):
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
    print(f"Subjects processed: {list(filtered.keys())}")

    return filtered

# English:
# Purpose:
# Define the Optuna objective function for hyperparameter optimization
# using Leave-One-Subject-Out (LOSO) cross-validation.
# The objective maximizes the mean macro F1-score across all subjects.
#
# Parameters:
# - trial: Optuna trial object used to sample hyperparameters.
# - all_subjects: Dictionary where keys are subject IDs and values are DataFrames.
# - label_col: Name of the target label column (default: "Action Label").
#
# Returns:
# - Mean macro F1-score across all LOSO folds.
#
# 日本語:
# 目的:
# Leave-One-Subject-Out（LOSO）交差検証を用いた
# Optunaのハイパーパラメータ最適化用目的関数を定義します。
# 全被験者の平均Macro F1-scoreを最大化します。
#
# パラメータ:
# - trial: ハイパーパラメータをサンプリングするためのOptuna Trialオブジェクト。
# - all_subjects: キーが被験者ID、値がDataFrameである辞書。
# - label_col: 目的ラベル列の名前（デフォルト: "Action Label"）。
#
# 戻り値:
# - 全LOSO分割における平均Macro F1-score。
def optuna_objective_loso(trial, all_subjects, label_col='Action Label'):
    # Sample a set of Random Forest hyperparameters.
    # Random Forestのハイパーパラメータをサンプリングする。
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth':         trial.suggest_categorical('max_depth', [5, 10, 20, 30, None]),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
        'random_state': 42,
        'n_jobs': 4
    }

    loso_folds = get_loso_folds(all_subjects)
    fold_scores = []

    for X_train, y_train, X_test, y_test, _, __ in loso_folds:

        model = model_random_forest(params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_scores.append(
            f1_score(y_test, preds, average='macro', zero_division=0)
        )

    return np.mean(fold_scores)


def run_optuna_study(all_subjects, n_trials=50, label_col='Action Label'):
    study = optuna.create_study(
        direction='maximize',
        study_name='rf_loso_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print_clean_header(f"\nOPTUNA SEARCH  —  {n_trials} trials  |  4 LOSO folds each")

    # tqdm bar — one tick per completed trial
    progress_bar = tqdm(
        total=n_trials,
        desc='Tuning',
        unit='trial',
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} '
                   '[{elapsed}<{remaining}, {rate_fmt}]'
    )

    best_so_far = [0.0]  # list so the closure can mutate it

    def callback(study, trial):
        # Update best tracker
        if trial.value > best_so_far[0]:
            best_so_far[0] = trial.value

        # Update the bar's suffix with live stats
        progress_bar.set_postfix({
            'current': f'{trial.value:.4f}',
            'best':    f'{best_so_far[0]:.4f}'
        })
        progress_bar.update(1)

    study.optimize(
        lambda trial: optuna_objective_loso(trial, all_subjects, label_col),
        n_trials=n_trials,
        callbacks=[callback]
    )

    progress_bar.close()

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'═' * 58}")
    print(f"  BEST RESULT")
    print(f"{'═' * 58}")
    print(f"  Mean LOSO Macro F1 : {study.best_value:.4f}")
    print(f"  Best trial number  : {study.best_trial.number}")
    print(f"\n  Best Parameters:")
    for param, value in study.best_params.items():
        print(f"    {param:<25} {value}")
    print(f"{'═' * 58}\n")

    return study