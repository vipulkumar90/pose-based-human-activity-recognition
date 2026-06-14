from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from har_utils.config import (
    ADABOOST_PARAMS,
    AGGLOMERATIVE_PARAMS,
    DBSCAN_PARAMS,
    DECISION_TREE_PARAMS,
    EXTRA_TREES_PARAMS,
    GAUSSIAN_NB_PARAMS,
    HIST_GRADIENT_PARAMS,
    KMEANS_PARAMS, 
    KNN_PARAMS, 
    LOGISTIC_REGRESSION_PARAMS, 
    RANDOM_FOREST_PARAMS, 
    SVM_PARAMS, 
    XGBOOST_PARAMS
)

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
def model_random_forest(hyperparams=None):
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
def model_hist_gradient(hyperparams=None):
    hyperparams = hyperparams or HIST_GRADIENT_PARAMS
    return HistGradientBoostingClassifier(**hyperparams)

# English:
# Purpose:
# Create a LogisticRegression classifier using either supplied hyperparameters
# or the default LOGISTIC_REGRESSION_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of classifier hyperparameters.
#
# Returns:
# - Configured LogisticRegression instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの LOGISTIC_REGRESSION_PARAMS を使用して
# LogisticRegression を生成します。
#
# パラメータ:
# - hyperparams: オプションの分類器ハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの LogisticRegression インスタンス。
def model_logistic_regression(hyperparams=None):
    hyperparams = hyperparams or LOGISTIC_REGRESSION_PARAMS
    return LogisticRegression(**hyperparams)

# English:
# Purpose:
# Create a KNeighborsClassifier using either supplied hyperparameters
# or the default KNN_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of classifier hyperparameters.
#
# Returns:
# - Configured KNeighborsClassifier instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの KNN_PARAMS を使用して
# KNeighborsClassifier を生成します。
#
# パラメータ:
# - hyperparams: オプションの分類器ハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの KNeighborsClassifier インスタンス。
def model_knn(hyperparams=None):
    hyperparams = hyperparams or KNN_PARAMS
    return KNeighborsClassifier(**hyperparams)

# English:
# Purpose:
# Create an XGBClassifier using either supplied hyperparameters
# or the default XGBOOST_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of classifier hyperparameters.
#
# Returns:
# - Configured XGBClassifier instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの XGBOOST_PARAMS を使用して
# XGBClassifier を生成します。
#
# パラメータ:
# - hyperparams: オプションの分類器ハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの XGBClassifier インスタンス。
def model_xgboost(hyperparams=None):
    hyperparams = hyperparams or XGBOOST_PARAMS
    return XGBClassifier(**hyperparams)

# English:
# Purpose:
# Create an SVC classifier using either supplied hyperparameters
# or the default SVM_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of classifier hyperparameters.
#
# Returns:
# - Configured SVC instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの SVM_PARAMS を使用して
# SVC を生成します。
#
# パラメータ:
# - hyperparams: オプションの分類器ハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの SVC インスタンス。
def model_svm(hyperparams=None):
    hyperparams = hyperparams or SVM_PARAMS
    return SVC(**hyperparams)

# English:
# Purpose:
# Create a DecisionTreeClassifier using either supplied hyperparameters
# or the default DECISION_TREE_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of classifier hyperparameters.
#
# Returns:
# - Configured DecisionTreeClassifier instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの DECISION_TREE_PARAMS を使用して
# DecisionTreeClassifier を生成します。
#
# パラメータ:
# - hyperparams: オプションの分類器ハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの DecisionTreeClassifier インスタンス。
def model_decision_tree(hyperparams=None):
    hyperparams = hyperparams or DECISION_TREE_PARAMS
    return DecisionTreeClassifier(**hyperparams)

# English:
# Purpose:
# Create an ExtraTreesClassifier using either supplied hyperparameters
# or the default EXTRA_TREES_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of classifier hyperparameters.
#
# Returns:
# - Configured ExtraTreesClassifier instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの EXTRA_TREES_PARAMS を使用して
# ExtraTreesClassifier を生成します。
#
# パラメータ:
# - hyperparams: オプションの分類器ハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの ExtraTreesClassifier インスタンス。
def model_extra_trees(hyperparams=None):
    hyperparams = hyperparams or EXTRA_TREES_PARAMS
    return ExtraTreesClassifier(**hyperparams)

# English:
# Purpose:
# Create an AdaBoostClassifier using either supplied hyperparameters
# or the default ADABOOST_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of classifier hyperparameters.
#
# Returns:
# - Configured AdaBoostClassifier instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの ADABOOST_PARAMS を使用して
# AdaBoostClassifier を生成します。
#
# パラメータ:
# - hyperparams: オプションの分類器ハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの AdaBoostClassifier インスタンス。
def model_adaboost(hyperparams=None):
    hyperparams = hyperparams or ADABOOST_PARAMS
    return AdaBoostClassifier(**hyperparams)

# English:
# Purpose:
# Create a GaussianNB classifier using either supplied hyperparameters
# or the default GAUSSIAN_NB_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of classifier hyperparameters.
#
# Returns:
# - Configured GaussianNB instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの GAUSSIAN_NB_PARAMS を使用して
# GaussianNB を生成します。
#
# パラメータ:
# - hyperparams: オプションの分類器ハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの GaussianNB インスタンス。
def model_gaussian_nb(hyperparams=None):
    hyperparams = hyperparams or GAUSSIAN_NB_PARAMS
    return GaussianNB(**hyperparams)

# English:
# Purpose:
# Create a KMeans clustering model using either supplied hyperparameters
# or the default KMEANS_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of clustering hyperparameters.
#
# Returns:
# - Configured KMeans instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの KMEANS_PARAMS を使用して
# KMeans を生成します。
#
# パラメータ:
# - hyperparams: オプションのクラスタリングハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの KMeans インスタンス。
def model_kmeans(hyperparams=None):
    hyperparams = hyperparams or KMEANS_PARAMS
    return KMeans(**hyperparams)

# English:
# Purpose:
# Create a DBSCAN clustering model using either supplied hyperparameters
# or the default DBSCAN_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of clustering hyperparameters.
#
# Returns:
# - Configured DBSCAN instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの DBSCAN_PARAMS を使用して
# DBSCAN を生成します。
#
# パラメータ:
# - hyperparams: オプションのクラスタリングハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの DBSCAN インスタンス。
def model_dbscan(hyperparams=None):
    hyperparams = hyperparams or DBSCAN_PARAMS
    return DBSCAN(**hyperparams)

# English:
# Purpose:
# Create an AgglomerativeClustering model using either supplied hyperparameters
# or the default AGGLOMERATIVE_PARAMS.
#
# Parameters:
# - hyperparams: Optional dict of clustering hyperparameters.
#
# Returns:
# - Configured AgglomerativeClustering instance.
#
# 日本語:
# 目的:
# 提供されたハイパーパラメータを使用するか、
# またはデフォルトの AGGLOMERATIVE_PARAMS を使用して
# AgglomerativeClustering を生成します。
#
# パラメータ:
# - hyperparams: オプションのクラスタリングハイパーパラメータ辞書。
#
# 戻り値:
# - 設定済みの AgglomerativeClustering インスタンス。
def model_agglomerative(hyperparams=None):
    hyperparams = hyperparams or AGGLOMERATIVE_PARAMS
    return AgglomerativeClustering(**hyperparams)

