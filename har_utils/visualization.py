import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import math

def plot_compare_feature(
    original_df,
    transformed_df,
    feature,
    subject_name,
    start=None,
    end=None
):
    """
    Compare a feature before and after normalization.

    Parameters
    ----------
    original_df : pd.DataFrame
        Original dataframe

    transformed_df : pd.DataFrame
        Normalized/transformed dataframe

    feature : str
        Column name to visualize

    start : int, optional
        Starting row index

    end : int, optional
        Ending row index
    """

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
    
# ==========================================

def plot_compare_feature_side_by_side(
    original_df,
    transformed_df,
    feature,
    subject_name,
    start=None,
    end=None
):
    """
    Compare a feature before and after transformation
    using separate axes.
    """

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

# ===============================================
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

def plot_class_distribution(subject, subject_name="No Subject Name Provided"):
    labels = subject['Action Label'].fillna('Unlabelled')
    labels.value_counts().plot(kind='barh')
    
    plt.title(f'Class Distribution: {subject_name}')
    plt.xlabel('Number of Frames')
    plt.ylabel('Activity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_percentage_distribution(subject, subject_name="No Subject Name Provided"):
    # Percentage Distribution
    class_pct = (
        subject['Action Label']
            .fillna('Unlabelled')
            .value_counts(normalize=True)
            .mul(100)
    )
    
    plt.figure(figsize=(6, 6))
    
    wedges, texts, autotexts = plt.pie(
        class_pct,
        autopct='%1.1f%%',
        startangle=90
    )
    
    plt.legend(
        wedges,
        class_pct.index,
        title="Activities",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8
    )
    
    plt.title(f"Activity Distribution (%): {subject_name}")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, prediction, title="Normalized Confusion Matrix"):
    cm = confusion_matrix(y_test, prediction, normalize='true')

    plt.figure(figsize=(10, 8))
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