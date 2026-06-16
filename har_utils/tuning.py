import numpy as np
import optuna
from sklearn.metrics import f1_score
from tqdm import tqdm

from har_utils.analysis import get_loso_folds, prepare_labels
from har_utils.data import print_clean_header

# English:
# Purpose:
# Define a generic Optuna objective function for hyperparameter optimization
# using Leave-One-Subject-Out (LOSO) cross-validation.
# The objective maximizes the mean Macro F1-score across all subjects,
# regardless of the underlying supervised learning model.
#
# Parameters:
# - trial: Optuna Trial object used to sample hyperparameters.
# - all_subjects: Dictionary where keys are subject IDs and values are DataFrames.
# - model_builder: Function that constructs and returns a model instance
#                  from a dictionary of hyperparameters.
# - param_sampler: Function that samples hyperparameters from the Optuna trial
#                  and returns them as a dictionary.
#
# Returns:
# - Mean Macro F1-score across all LOSO folds.
#
# 日本語:
# 目的:
# Leave-One-Subject-Out（LOSO）交差検証を用いた
# 汎用的なOptunaハイパーパラメータ最適化用目的関数を定義します。
# 使用する教師あり学習モデルに関係なく、
# 全被験者の平均Macro F1-scoreを最大化します。
#
# パラメータ:
# - trial: ハイパーパラメータをサンプリングするためのOptuna Trialオブジェクト。
# - all_subjects: キーが被験者ID、値がDataFrameである辞書。
# - model_builder: ハイパーパラメータ辞書からモデルを生成する関数。
# - param_sampler: Optuna Trialからハイパーパラメータを
#                  サンプリングして辞書として返す関数。
#
# 戻り値:
# - 全LOSO分割における平均Macro F1-score。
def optuna_objective_loso(
    trial,
    all_subjects,
    model_builder,
    param_sampler
):
    # Sample hyperparameters for the selected model.
    # 選択したモデルのハイパーパラメータをサンプリングする。
    params = param_sampler(trial)

    loso_folds = get_loso_folds(all_subjects)
    fold_scores = []

    for X_train, y_train, X_test, y_test, _, __ in loso_folds:

        # Build the model using the sampled hyperparameters.
        # サンプリングしたハイパーパラメータでモデルを生成する。
        model = model_builder(params)

        # Encode the labels in case of XGBoost
        y_train, y_test = prepare_labels(model, y_train, y_test)
        
        # Train the model on the training fold.
        # 学習データでモデルを学習する。
        model.fit(X_train, y_train)

        # Predict labels for the test fold.
        # テストデータのラベルを予測する。
        preds = model.predict(X_test)

        # Compute Macro F1-score for the current fold.
        # 現在のフォールドのMacro F1-scoreを計算する。
        fold_scores.append(
            f1_score(
                y_test,
                preds,
                average='macro',
                zero_division=0
            )
        )

    # Return the mean score across all LOSO folds.
    # 全LOSOフォールドの平均スコアを返す。
    return np.mean(fold_scores)

# English:
# Purpose:
# Run an Optuna hyperparameter optimization study using
# Leave-One-Subject-Out (LOSO) cross-validation.
# The study can optimize any supervised learning model by supplying
# a model builder function and a corresponding hyperparameter sampler.
#
# Parameters:
# - all_subjects: Dictionary where keys are subject IDs and values are DataFrames.
# - model_builder: Function that constructs a model from a hyperparameter dictionary.
# - param_sampler: Function that samples hyperparameters from an Optuna trial.
# - n_trials: Number of Optuna trials to perform (default: 50).
#
# Returns:
# - Completed Optuna Study object containing the optimization results.
#
# 日本語:
# 目的:
# Leave-One-Subject-Out（LOSO）交差検証を用いて
# Optunaによるハイパーパラメータ最適化を実行します。
# モデル生成関数とハイパーパラメータサンプリング関数を渡すことで、
# 任意の教師あり学習モデルに対応できます。
#
# パラメータ:
# - all_subjects: キーが被験者ID、値がDataFrameである辞書。
# - model_builder: ハイパーパラメータ辞書からモデルを生成する関数。
# - param_sampler: Optuna Trialからハイパーパラメータを
#                  サンプリングする関数。
# - n_trials: Optunaの試行回数（デフォルト: 50）。
#
# 戻り値:
# - 最適化結果を保持するOptuna Studyオブジェクト。
def run_optuna_study(
    all_subjects,
    model_builder,
    param_sampler,
    n_trials=50
):
    study = optuna.create_study(
        direction='maximize',
        study_name=f'{model_builder.__name__}_loso_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print_clean_header(
        f"\nOPTUNA SEARCH — {model_builder.__name__} | "
        f"{n_trials} trials | 4 LOSO folds each"
    )

    # Create a progress bar for completed Optuna trials.
    # Optuna試行の進行状況を表示するプログレスバーを作成する。
    progress_bar = tqdm(
        total=n_trials,
        desc='Tuning',
        unit='trial',
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} '
                   '[{elapsed}<{remaining}, {rate_fmt}]'
    )

    # Track the best score found during optimization.
    # 最適化中の最高スコアを記録する。
    best_so_far = [0.0]

    def callback(study, trial):

        # Update the best score if the current trial improves it.
        # 現在の試行が最高スコアを更新した場合は記録する。
        if trial.value > best_so_far[0]:
            best_so_far[0] = trial.value

        # Update progress bar statistics.
        # プログレスバーの情報を更新する。
        progress_bar.set_postfix({
            'current': f'{trial.value:.4f}',
            'best': f'{best_so_far[0]:.4f}'
        })

        progress_bar.update(1)

    # Run the Optuna optimization.
    # Optunaによる最適化を実行する。
    study.optimize(
        lambda trial: optuna_objective_loso(
            trial,
            all_subjects,
            model_builder,
            param_sampler
        ),
        n_trials=n_trials,
        callbacks=[callback]
    )

    progress_bar.close()

    # Print a summary of the optimization results.
    # 最適化結果の概要を表示する。
    print(f"\n{'═' * 58}")
    print("  BEST RESULT")
    print(f"{'═' * 58}")
    print(f"  Mean LOSO Macro F1 : {study.best_value:.4f}")
    print(f"  Best trial number  : {study.best_trial.number}")
    print("\n  Best Parameters:")
    for param, value in study.best_params.items():
        print(f"    {param:<25} {value}")
    print(f"{'═' * 58}\n")

    return study

 # English:
# Purpose:
# Sample a set of XGBoost hyperparameters from the Optuna search space.
# The sampled parameters are used to construct an XGBClassifier during
# hyperparameter optimization.
#
# Parameters:
# - trial: Optuna Trial object used to sample hyperparameters.
#
# Returns:
# - Dictionary of sampled XGBoost hyperparameters.
#
# 日本語:
# 目的:
# Optunaの探索空間からXGBoostのハイパーパラメータをサンプリングします。
# サンプリングされたパラメータは、ハイパーパラメータ最適化時に
# XGBClassifierを生成するために使用されます。
#
# パラメータ:
# - trial: ハイパーパラメータをサンプリングするための
#          Optuna Trialオブジェクト。
#
# 戻り値:
# - サンプリングされたXGBoostハイパーパラメータの辞書。
def sample_xgb_params(trial):

    # Sample a set of XGBoost hyperparameters.
    # XGBoostのハイパーパラメータをサンプリングする。
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 500, step=50
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 10
        ),
        "gamma": trial.suggest_float(
            "gamma", 0.0, 5.0
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-4, 10.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-4, 10.0, log=True
        ),
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": 4
    }

# English:
# Purpose:
# Sample a set of Random Forest hyperparameters from the Optuna search space.
# The sampled parameters are used to construct a RandomForestClassifier during
# hyperparameter optimization.
#
# Parameters:
# - trial: Optuna Trial object used to sample hyperparameters.
#
# Returns:
# - Dictionary of sampled Random Forest hyperparameters.
#
# 日本語:
# 目的:
# Optunaの探索空間からRandom Forestのハイパーパラメータをサンプリングします。
# サンプリングされたパラメータは、ハイパーパラメータ最適化時に
# RandomForestClassifierを生成するために使用されます。
#
# パラメータ:
# - trial: ハイパーパラメータをサンプリングするための
#          Optuna Trialオブジェクト。
#
# 戻り値:
# - サンプリングされたRandom Forestハイパーパラメータの辞書。
def sample_rf_params(trial):

    # Sample a set of Random Forest hyperparameters.
    # Random Forestのハイパーパラメータをサンプリングする。
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 500, step=50
        ),
        "max_depth": trial.suggest_categorical(
            "max_depth", [5, 10, 20, 30, None]
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 1, 30
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 20
        ),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5]
        ),
        "random_state": 42,
        "n_jobs": 4
    }