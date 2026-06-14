import pandas as pd
from sklearn.model_selection import train_test_split
from har_utils.config import BASE_FILE_NAME, BASE_DIR, FILE_TYPE
import random


# English:
# Purpose: Load a CSV file for a specific subject and return it as a DataFrame.
# Parameters:
# - suffix: Subject-specific identifier appended to the base file name.
# Returns:
# - DataFrame loaded from the CSV file with the first column as index.
# 日本語:
# 目的: 特定の被験者に対応するCSVファイルを読み込み、DataFrameとして返します。
# パラメータ:
# - suffix: 基本ファイル名に付加される被験者固有の識別子。
# 戻り値:
# - 先頭列をインデックスとして使用したCSV読み込み結果のDataFrame。
def get_subject(suffix):
    return pd.read_csv(f"{BASE_DIR}/{BASE_FILE_NAME}_{suffix}{FILE_TYPE}", header=0, index_col=0)


# English:
# Purpose: Print a stylized header to the console for better readability.
# Parameters:
# - title: Text displayed in the center of the header.
# - line_len: Total width of the header line.
# - border_char: Character used to draw the top and bottom border lines.
# Returns:
# - None. Side effect: Writes a formatted header to stdout.
#
# 日本語:
# 目的: コンソールの可読性を高めるために装飾されたヘッダーを表示します。
# パラメータ:
# - title: ヘッダー中央に表示されるテキスト。
# - line_len: ヘッダー行の全体幅。
# - border_char: ヘッダー上下の罫線を描画するために使用する文字。
# 戻り値:
# - なし。副作用: 標準出力に整形済みヘッダーを出力します。
def print_clean_header(title='No Title', line_len=80, border_char='='):
    print(f"\n{border_char * line_len}")
    print(f"{title:^{line_len}}")
    print(f"{border_char * line_len}")

# English:
# Purpose: Partition subject data by selecting one random subject for testing.
# Parameters:
# - all_subject: Mapping of subject suffixes to their DataFrames.
# - random_state: Seed to make the random selection reproducible.
# - show_shape: If true, prints the shapes of train and test splits.
# Returns:
# - A tuple of (train, test) DataFrames.
# 日本語:
# 目的: 被験者データを分割し、ランダムに1つの被験者をテスト用に選択します。
# パラメータ:
# - all_subject: 被験者サフィックスとDataFrameのマッピング。
# - random_state: ランダム選択を再現可能にするためのシード。
# - show_shape: true の場合、train/test の形状を表示します。
# 戻り値:
# - (train, test) のタプル。
def get_train_test_split(all_subject, random_state=42, show_shape=False):
    rng = random.Random(random_state)
    subject_suffix_list = list(all_subject.keys())

    # Choose a single subject at random for the test split.
    # テスト分割のために1つの被験者をランダムに選択します。
    test_key = rng.choice(subject_suffix_list)
    train_key = [s for s in subject_suffix_list if s != test_key]
    
    train = pd.concat([all_subject[s] for s in train_key], ignore_index=True)
    test = all_subject[test_key]
    if show_shape:
        print(f"Train Shape:{train.shape}")
        print(f"Test Shape:{test.shape}")

    print(f"Train Subjects: {train_key}")
    print(f"Test Subject : {test_key}")
    
    return train, test

# English:
# Purpose: Convert train/test DataFrames into feature matrices and label vectors.
# Parameters:
# - train: Training DataFrame containing features and labels.
# - test: Test DataFrame containing features and labels.
# - show_shape: If true, prints the shapes of the resulting splits.
# Returns:
# - Tuple of (X_train, y_train, X_test, y_test).
# 日本語:
# 目的: train/test の DataFrame を特徴量とラベルの分割に変換します。
# パラメータ:
# - train: 特徴量とラベルを含む学習用DataFrame。
# - test: 特徴量とラベルを含むテスト用DataFrame。
# - show_shape: true の場合、生成された分割の形状を表示します。
# 戻り値:
# - (X_train, y_train, X_test, y_test) のタプル。
def get_X_y_split_base(train, test, show_shape=False):
    X_train = train.drop(columns=['Action Label'])
    y_train = train['Action Label']
    X_test = test.drop(columns=['Action Label'])
    y_test = test['Action Label']

    if show_shape:
        print(f"Shape of X_train:{X_train.shape}")
        print(f"Shape of X_test:{X_test.shape}")
        print(f"Shape of y_train:{y_train.shape}")
        print(f"Shape of y_test:{y_test.shape}")
        
    return X_train, y_train, X_test, y_test

# English:
# Purpose: Build a train/test split and then derive feature/label pairs.
# Parameters:
# - all_subject: Mapping of subject suffixes to DataFrames.
# - random_state: Seed used by the train/test split.
# - show_shape: If true, forwards the display flag to the base split function.
# Returns:
# - The output of get_X_y_split_base: (X_train, y_train, X_test, y_test).
# 日本語:
# 目的: 学習/テスト分割を作成し、特徴量/ラベルのペアを生成します。
# パラメータ:
# - all_subject: 被験者サフィックスとDataFrameのマッピング。
# - random_state: train/test 分割で使用するシード。
# - show_shape: true の場合、表示フラグを基礎分割関数に渡します。
# 戻り値:
# - get_X_y_split_base の出力: (X_train, y_train, X_test, y_test)。
def get_X_y_split(all_subject, random_state=42, show_shape=False):
    if len(all_subject) < 2:
        raise ValueError("At least 2 subjects are required for train/test split.")
    train, test = get_train_test_split(all_subject, random_state)
    return get_X_y_split_base(train, test, show_shape)

# English:
# Purpose:
# Split a single subject dataset into training and testing sets.
# The feature matrix and target labels are extracted automatically,
# and an optional stratified split is performed.
#
# Parameters:
# - subject: DataFrame containing features and the Action Label column.
# - label_column: Name of the target label column (default: "Action Label").
# - test_size: Fraction of the dataset reserved for testing (default: 0.2).
# - random_state: Random seed for reproducible splitting (default: 42).
# - stratify: Whether to preserve class distribution during splitting (default: True).
#
# Returns:
# - X_train: Training feature matrix.
# - y_train: Training labels.
# - X_test: Testing feature matrix.
# - y_test: Testing labels.
#
# 日本語:
# 目的:
# 単一被験者のデータセットを学習用データとテスト用データに分割します。
# 特徴量と目的ラベルを自動的に抽出し、
# 必要に応じて層化抽出（Stratified Split）を行います。
#
# パラメータ:
# - subject: 特徴量と Action Label 列を含むDataFrame。
# - label_column: 目的ラベル列の名前（デフォルト: "Action Label"）。
# - test_size: テストデータとして使用する割合（デフォルト: 0.2）。
# - random_state: 再現性を確保するための乱数シード（デフォルト: 42）。
# - stratify: クラス分布を維持してデータを分割するかどうか（デフォルト: True）。
#
# 戻り値:
# - X_train: 学習用特徴量。
# - y_train: 学習用ラベル。
# - X_test: テスト用特徴量。
# - y_test: テスト用ラベル。
def get_X_y_split_single_subject(
    subject,
    label_column="Action Label",
    test_size=0.2,
    random_state=42,
    stratify=True
):
    X = subject.drop(columns=[label_column])
    y = subject[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )

    return X_train, y_train, X_test, y_test