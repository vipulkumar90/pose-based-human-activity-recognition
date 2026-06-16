import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import math
import pandas as pd
import numpy as np

# English:
# Purpose: Plot a single feature before and after transformation on the same axes with overlaid lines.
# Parameters:
# - original_df: DataFrame with original (untransformed) values.
# - transformed_df: DataFrame with normalized/transformed values.
# - feature: Column name to visualize.
# - subject_name: Label for the plot title.
# - start: Starting row index for slicing (default: None, use entire range).
# - end: Ending row index for slicing (default: None, use entire range).
# Returns:
# - None. Side effect: Displays matplotlib plot to screen.
# 日本語:
# 目的: 変換前後の単一特徴量を同じ軸に重ね描きします。
# パラメータ:
# - original_df: 元の値を含むDataFrame。
# - transformed_df: 正規化/変換後の値を含むDataFrame。
# - feature: 可視化する列名。
# - subject_name: プロットタイトルのラベル。
# - start: スライス用の開始行インデックス（デフォルト: None、全範囲使用）。
# - end: スライス用の終了行インデックス（デフォルト: None、全範囲使用）。
# 戻り値:
# - なし。副作用: matplotlibプロットを画面に表示します。
def plot_compare_feature(
    original_df,
    transformed_df,
    feature,
    subject_name,
    start=None,
    end=None
):

    plt.figure(figsize=(12, 5))

    plt.plot(
        original_df[feature].iloc[start:end],
        label="Original",
        alpha=0.8
    )

    plt.plot(
        transformed_df[feature].iloc[start:end],
        label="Normalized",
        alpha=0.8
    )

    plt.title(f"{subject_name} - {feature}: Before vs After")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# English:
# Purpose: Plot before and after feature transformation on separate side-by-side axes.
# Parameters:
# - original_df: DataFrame with original (untransformed) values.
# - transformed_df: DataFrame with normalized/transformed values.
# - feature: Column name to visualize.
# - subject_name: Label for the plot title.
# - start: Starting row index for slicing (default: None, use entire range).
# - end: Ending row index for slicing (default: None, use entire range).
# Returns:
# - None. Side effect: Displays matplotlib figure with 1x2 subplots to screen.
# 日本語:
# 目的: 変換前後の特徴量を並んだ個別の軸に描画します。
# パラメータ:
# - original_df: 元の値を含むDataFrame。
# - transformed_df: 正規化/変換後の値を含むDataFrame。
# - feature: 可視化する列名。
# - subject_name: プロットタイトルのラベル。
# - start: スライス用の開始行インデックス（デフォルト: None、全範囲使用）。
# - end: スライス用の終了行インデックス（デフォルト: None、全範囲使用）。
# 戻り値:
# - なし。副作用: 1x2サブプロット付きmatplotlibフィギュアを画面に表示します。
def plot_compare_feature_side_by_side(
    original_df,
    transformed_df,
    feature,
    subject_name,
    start=None,
    end=None
):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5)
    )

    # Original
    axes[0].plot(
        original_df[feature].iloc[start:end]
    )

    axes[0].set_title(
        f"Original {subject_name}: {feature}"
    )

    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.3)

    # Normalized
    axes[1].plot(
        transformed_df[feature].iloc[start:end]
    )

    axes[1].set_title(
        f"Normalized {subject_name}: {feature}"
    )

    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Value")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# English:
# Purpose: Create side-by-side histograms of feature distributions before and after transformation.
# Parameters:
# - original_df: DataFrame with original (untransformed) feature values.
# - transformed_df: DataFrame with normalized/transformed feature values.
# - feature: Column name of the feature to visualize.
# - bins: Number of histogram bins (default: 50).
# Returns:
# - None. Side effect: Displays matplotlib figure with two histograms to screen.
# 日本語:
# 目的: 変換前後の特徴量分布を並んだヒストグラムで描画します。
# パラメータ:
# - original_df: 元の特徴量値を含むDataFrame。
# - transformed_df: 正規化/変換後の特徴量値を含むDataFrame。
# - feature: 可視化する特徴量の列名。
# - bins: ヒストグラムのビン数（デフォルト: 50）。
# 戻り値:
# - なし。副作用: 2つのヒストグラムを含むmatplotlibフィギュアを画面に表示します。
def plot_compare_histograms(
    original_df,
    transformed_df,
    feature,
    bins=50
):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5)
    )

    axes[0].hist(
        original_df[feature],
        bins=bins
    )

    axes[0].set_title(
        f"Original Distribution\n{feature}"
    )

    axes[1].hist(
        transformed_df[feature],
        bins=bins
    )

    axes[1].set_title(
        f"Normalized Distribution\n{feature}"
    )

    plt.tight_layout()
    plt.show()

# English:
# Purpose:
# Plot the activity class distribution for one or more subjects.
# When multiple subjects are provided, the function automatically
# arranges the plots into a suitable subplot grid.
#
# Parameters:
# - data:
#     * A single subject DataFrame, or
#     * A dictionary where keys are subject IDs and values are DataFrames.
#
# Returns:
# - None. Side effect: Displays one or more horizontal bar charts.
#
# 日本語:
# 目的:
# 1人または複数被験者の活動クラス分布を表示します。
# 複数被験者が入力された場合は、自動的に適切なサブプロット配置で
# 水平棒グラフを表示します。
#
# パラメータ:
# - data:
#     * 単一被験者のDataFrame
#     * またはキーが被験者ID、値がDataFrameの辞書
#
# 戻り値:
# - なし。副作用: 活動クラス分布の水平棒グラフを表示します。
def plot_class_distribution(data):

    import math

    # Handle a single subject DataFrame.
    # 単一被験者のDataFrameを処理する。
    if isinstance(data, pd.DataFrame):
        subjects = {"Subject": data}

    # Handle multiple subjects stored in a dictionary.
    # 辞書に格納された複数被験者を処理する。
    elif isinstance(data, dict):
        subjects = data

    else:
        raise TypeError(
            "Expected a pandas DataFrame or a dictionary of DataFrames."
        )

    n_subjects = len(subjects)

    # Determine an appropriate subplot layout.
    # 被験者数に応じてサブプロット配置を決定する。
    if n_subjects == 1:
        rows, cols = 1, 1
    elif n_subjects <= 3:
        rows, cols = 1, n_subjects
    elif n_subjects <= 4:
        rows, cols = 2, 2
    else:
        cols = 3
        rows = math.ceil(n_subjects / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6 * cols, 4 * rows)
    )

    if n_subjects == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for ax, (subject_name, subject) in zip(axes, subjects.items()):

        labels = subject["Action Label"].fillna("Unlabelled")
        counts = labels.value_counts()

        ax.barh(counts.index, counts.values)

        ax.set_title(f"{subject_name}")
        ax.set_xlabel("Number of Frames")
        ax.set_ylabel("Activity")

    # Remove any unused subplot axes.
    # 使用されなかったサブプロットを削除する。
    for ax in axes[n_subjects:]:
        fig.delaxes(ax)

    fig.suptitle("Class Distribution", fontsize=16)

    plt.tight_layout()
    plt.show()

# English:
# Purpose:
# Plot the percentage distribution of activity classes for one or more subjects.
# When multiple subjects are provided, the function automatically
# arranges the plots into a suitable subplot grid.
#
# Parameters:
# - data:
#     * A single subject DataFrame, or
#     * A dictionary where keys are subject IDs and values are DataFrames.
#
# Returns:
# - None. Side effect: Displays one or more pie charts.
#
# 日本語:
# 目的:
# 1人または複数被験者の活動クラス割合を表示します。
# 複数被験者が入力された場合は、自動的に適切なサブプロット配置で
# 円グラフを表示します。
#
# パラメータ:
# - data:
#     * 単一被験者のDataFrame
#     * またはキーが被験者ID、値がDataFrameの辞書
#
# 戻り値:
# - なし。副作用: 活動クラス割合の円グラフを表示します。
def plot_percentage_distribution(data):

    import math

    # Handle a single subject DataFrame.
    # 単一被験者のDataFrameを処理する。
    if isinstance(data, pd.DataFrame):
        subjects = {"Subject": data}

    # Handle multiple subjects stored in a dictionary.
    # 辞書に格納された複数被験者を処理する。
    elif isinstance(data, dict):
        subjects = data

    else:
        raise TypeError(
            "Expected a pandas DataFrame or a dictionary of DataFrames."
        )

    n_subjects = len(subjects)

    # Determine an appropriate subplot layout.
    # 被験者数に応じてサブプロット配置を決定する。
    if n_subjects == 1:
        rows, cols = 1, 1
    elif n_subjects <= 3:
        rows, cols = 1, n_subjects
    elif n_subjects <= 4:
        rows, cols = 2, 2
    else:
        cols = 3
        rows = math.ceil(n_subjects / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6 * cols, 5 * rows)
    )

    if n_subjects == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for ax, (subject_name, subject) in zip(axes, subjects.items()):

        class_pct = (
            subject["Action Label"]
            .fillna("Unlabelled")
            .value_counts(normalize=True)
            .mul(100)
        )

        wedges, texts, autotexts = ax.pie(
            class_pct.values,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.legend(
            wedges,
            class_pct.index,
            title="Activities",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8
        )

        ax.set_title(f'{subject_name}')

    # Remove any unused subplot axes.
    # 使用されなかったサブプロットを削除する。
    for ax in axes[n_subjects:]:
        fig.delaxes(ax)

    fig.suptitle("Activity Distribution (%)", fontsize=16)

    plt.tight_layout()
    plt.show()

# English:
# Purpose: Create and display a normalized confusion matrix heatmap from predictions and true labels.
# Parameters:
# - y_test: True labels for the test set.
# - prediction: Predicted labels from the classifier.
# - title: Title for the heatmap (default: 'Normalized Confusion Matrix').
# Returns:
# - None. Side effect: Displays normalized confusion matrix heatmap to screen.
# 日本語:
# 目的: 予測値と真のラベルから正規化混同行列ヒートマップを作成・表示します。
# パラメータ:
# - y_test: テストセットの真のラベル。
# - prediction: 分類器からの予測ラベル。
# - title: ヒートマップのタイトル（デフォルト: 'Normalized Confusion Matrix'）。
# 戻り値:
# - なし。副作用: 正規化混同行列ヒートマップを画面に表示します。
def plot_confusion_matrix(y_test, prediction, title="Normalized Confusion Matrix"):
    cm = confusion_matrix(y_test, prediction, normalize='true')

    plt.figure(figsize=(8, 6))
    labels = sorted(y_test.unique())
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        square=True,
        linewidths=0.5,
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    
    plt.tight_layout()
    plt.show()

# English:
# Purpose: Create and save histograms of a feature across subjects for a specific activity class.
# Arranges plots in a grid, one subplot per subject.
# Parameters:
# - subjects_dict: Dictionary mapping subject names to DataFrames.
# - feature: Column name of the feature to visualize.
# - activity_class: Activity label value to filter for visualization.
# - label_col: Name of the activity label column (default: 'Action Label').
# - bins: Number of histogram bins per subplot (default: 40).
# - save_dir: Directory path for saving PNG file (default: 'feature_distribution_plots').
# Returns:
# - None. Side effect: Creates save_dir, generates feature__{class}.png, displays plot to screen.
# 日本語:
# 目的: 特定の活動クラスについて、被験者全体にわたる特徴量ヒストグラムを作成・保存します。
# プロット配置をグリッド上に配置し、被験者ごとに1つのサブプロット。
# パラメータ:
# - subjects_dict: 被験者名をDataFrameにマッピングする辞書。
# - feature: 可視化する特徴量の列名。
# - activity_class: フィルタリング対象のアクティビティラベル値。
# - label_col: アクティビティラベル列の名前（デフォルト: 'Action Label'）。
# - bins: サブプロットごとのヒストグラムビン数（デフォルト: 40）。
# - save_dir: PNG保存先ディレクトリパス（デフォルト: 'feature_distribution_plots'）。
# 戻り値:
# - なし。副作用: save_dirを作成し、feature__{class}.pngを生成して、画面に表示します。
def plot_feature_class_across_subjects(
    subjects_dict,
    feature,
    activity_class,
    label_col='Action Label',
    bins=40,
    save_dir='feature_distribution_plots'
):
    # Create output folder
    os.makedirs(save_dir, exist_ok=True)

    n_subjects = len(subjects_dict)

    # 2x2 grid for 4 subjects
    n_cols = 2
    n_rows = math.ceil(n_subjects / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 10),
        sharex=True
    )

    axes = axes.flatten()

    for ax, (subject_name, df) in zip(axes, subjects_dict.items()):

        class_data = df[df[label_col] == activity_class][feature]

        if class_data.empty:
            ax.text(
                0.5,
                0.5,
                'No data',
                ha='center',
                va='center'
            )
            ax.set_title(subject_name)
            continue

        ax.hist(
            class_data,
            bins=bins,
            edgecolor='white',
            linewidth=0.4
        )

        ax.set_title(
            f"{subject_name}\n(n={len(class_data)})"
        )
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[n_subjects:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Feature: '{feature}' | Class: '{activity_class}'",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    # Clean filename
    safe_feature = feature.replace(" ", "_")
    safe_class = activity_class.replace(" ", "_")

    filename = (
        f"{safe_feature}__{safe_class}.png"
    )

    filepath = os.path.join(
        save_dir,
        filename
    )

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close()