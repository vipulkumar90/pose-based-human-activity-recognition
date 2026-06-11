from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from har_utils.config import HIST_GRADIENT_PARAMS, RANDOM_FOREST_PARAMS

# English:
# Purpose:
# Create a RandomForestClassifier using either supplied hyperparameters
# or the default RANDOM_FOREST_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of classifier hyperparameters.
#
# Returns:
# - Configured RandomForestClassifier instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの RANDOM_FOREST_PARAMS を使用して RandomForestClassifier を生成します。
#
# パラメータ:
# - hyperparams: オプションの分類器ハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの RandomForestClassifier インスタンス。
def model_random_forest(hyperparams):
    hyperparams = hyperparams or RANDOM_FOREST_PARAMS
    return RandomForestClassifier(**hyperparams)

# English:
# Purpose:
# Create a HistGradientBoostingClassifier using either supplied hyperparameters
# or the default HIST_GRADIENT_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of classifier hyperparameters.
#
# Returns:
# - Configured HistGradientBoostingClassifier instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの HIST_GRADIENT_PARAMS を使用して
# HistGradientBoostingClassifier を生成します。
#
# パラメータ:
# - hyperparams: オプションの分類器ハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの HistGradientBoostingClassifier インスタンス。
def model_hist_gradient(hyperparams):
    hyperparams = hyperparams or HIST_GRADIENT_PARAMS
    return HistGradientBoostingClassifier(**hyperparams)