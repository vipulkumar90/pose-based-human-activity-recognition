import numpy as np
from har_utils.config import SHOULDER_WIDTH_THRESHOLD, FILE_NAME_SUFFIX
from har_utils.features import (
    add_angle_features,
    add_range_of_motion_features,
    add_acceleration_features,
    create_multiscale_windowed_features,
    create_windowed_features
)
from sklearn.preprocessing import StandardScaler


# English:
# Purpose:
# Generate a summary of common data quality issues for one or more subjects.
# The function checks for NaN values, infinite values, duplicate rows,
# and rows containing invalid numeric values.
#
# Parameters:
# - data: Either
#     * A single subject DataFrame, or
#     * A dictionary where keys are subject IDs and values are DataFrames.
# - subject_name: Name of the subject when a single DataFrame is supplied.
#                 Ignored when a dictionary is provided.
#
# Returns:
# - pd.DataFrame where each row summarizes the sanity check statistics
#   for one subject.
#
# 日本語:
# 目的:
# 1人または複数被験者のデータ品質を要約します。
# NaN値、Inf値、重複行、および無効な数値を含む行数を確認します。
#
# パラメータ:
# - data:
#     * 単一被験者のDataFrame
#     * またはキーが被験者ID、値がDataFrameの辞書
# - subject_name:
#     単一DataFrameの場合の被験者名。
#     辞書が渡された場合は無視されます。
#
# 戻り値:
# - 被験者ごとのサニティチェック結果をまとめたDataFrame。
def do_sanity_check(data, subject_name="Unknown"):

    def _summarize(df, name):

        numeric_df = df.select_dtypes(include=np.number)

        total_nan = df.isna().sum().sum()

        total_inf = np.isinf(numeric_df).sum().sum()

        duplicate_rows = df.duplicated().sum()

        rows_with_nan = df.isna().any(axis=1).sum()

        rows_with_inf = np.isinf(numeric_df).any(axis=1).sum()

        return {
            "Subject": name,
            "Rows": len(df),
            "Columns": df.shape[1],
            "NaN Count": total_nan,
            "Inf Count": total_inf,
            "Rows with NaN": rows_with_nan,
            "Rows with Inf": rows_with_inf,
            "Duplicate Rows": duplicate_rows,
        }

    # Handle a single subject DataFrame.
    # 単一被験者のDataFrameを処理する。
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame([
            _summarize(data, subject_name)
        ])

    # Handle multiple subjects stored in a dictionary.
    # 辞書に格納された複数被験者を処理する。
    elif isinstance(data, dict):
        return pd.DataFrame([
            _summarize(df, subject_id)
            for subject_id, df in data.items()
        ])

    # Raise an error for unsupported input types.
    # サポートされていない入力型の場合は例外を発生させる。
    else:
        raise TypeError(
            "Expected a pandas DataFrame or a dictionary of DataFrames."
        )

# ==========================================================
# Calculate shoulder width
# ==========================================================
# Why?
# Different subjects have different body sizes.
# People also stand closer or farther from the camera.
#
# Example:
# Person A shoulder width = 120 pixels
# Person B shoulder width = 180 pixels
#
# Even if they perform the same action,
# their raw coordinates will be very different.
#
# We use shoulder width as a scaling factor so
# all skeletons are represented at roughly the same scale.
# ==========================================================
# English:
# Purpose: Calculate Euclidean distance between left and right shoulders.
# Parameters:
# - subject: DataFrame with left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y columns.
# Returns:
# - Series of shoulder width values, one per frame.
# 日本語:
# 目的: 左肩と右肩間のユークリッド距離を計算します。
# パラメータ:
# - subject: 肩の座標列を含むDataFrame。
# 戻り値:
# - フレームごとの肩幅値のSeries。
def get_shoulder_width(subject):
    shoulder_width = np.sqrt(
        (subject['left_shoulder_x'] - subject['right_shoulder_x']) ** 2 +
        (subject['left_shoulder_y'] - subject['right_shoulder_y']) ** 2
    )
    
    return shoulder_width

# English:
# Purpose: Calculate Euclidean distance between torso endpoints (mid-shoulder to mid-hip).
# More stable scaling factor than shoulder width for skeleton normalization.
# Parameters:
# - subject: DataFrame with shoulder and hip keypoint coordinates.
# Returns:
# - Series of torso height values, one per frame.
# 日本語:
# 目的: 胴体エンドポイント（肩中心から腰中心）間のユークリッド距離を計算します。
# スケルトン正規化用のスケーリング係数として肩幅より安定しています。
# パラメータ:
# - subject: 肩と腰のキーポイント座標を含むDataFrame。
# 戻り値:
# - フレームごとの胴体高さ値のSeries。
def get_torso_height(subject):
    mid_shoulder_y = (subject['left_shoulder_y'] + subject['right_shoulder_y']) / 2
    mid_hip_y      = (subject['left_hip_y']      + subject['right_hip_y'])      / 2

    torso_height = np.sqrt(
        ((subject['left_shoulder_x'] + subject['right_shoulder_x']) / 2 -
         (subject['left_hip_x']      + subject['right_hip_x'])      / 2) ** 2 +
        (mid_shoulder_y - mid_hip_y) ** 2
    )
    return torso_height

# English:
# Purpose: Normalize skeleton keypoints to body-relative coordinates and scale invariant.
# Center at hip midpoint, scale by torso height or shoulder width for uniform skeleton representation.
# Parameters:
# - subject_original: DataFrame with raw keypoint coordinates (17 keypoints with _x, _y columns).
# - use_torso_height: If True, use torso height for scaling; else use shoulder width (default: True).
# Returns:
# - New DataFrame with normalized keypoint coordinates, same shape as input.
# 日本語:
# 目的: スケルトンキーポイントを身体相対座標にスケール不変な正規化します。
# 腰中点を中心に配置し、胴体高さまたは肩幅でスケーリングして均一なスケルトン表現を実現します。
# パラメータ:
# - subject_original: 生のキーポイント座標を含むDataFrame（17キーポイント）。
# - use_torso_height: Trueの場合は胴体高さでスケーリング、Falseの場合は肩幅を使用（デフォルト: True）。
# 戻り値:
# - 正規化されたキーポイント座標を含む新しいDataFrame、入力と同じ形状。
def modify_skeleton_normalization(subject_original, use_torso_height=True):
    
    # Create a copy
    subject_transformed = subject_original.copy()
    
    # ==========================================================
    # Calculate hip center
    # ==========================================================
    # Why?
    # Raw keypoints are recorded in image coordinates.
    # If a person moves left/right in the camera frame,
    # every x-coordinate changes even though the pose is identical.
    #
    # By subtracting the hip center, we convert coordinates
    # from "camera coordinates" to "body-relative coordinates".
    #
    # After this transformation:
    # (0,0) represents the center of the person's body.
    # ==========================================================
    
    hip_center_x = (
        subject_transformed['left_hip_x'] +
        subject_transformed['right_hip_x']
    ) / 2
    
    hip_center_y = (
        subject_transformed['left_hip_y'] +
        subject_transformed['right_hip_y']
    ) / 2

    # shoulder_width = get_shoulder_width(subject_transformed)

    shoulder_width = get_shoulder_width(subject_transformed)
    torso_height   = get_torso_height(subject_transformed)

    # Use torso height as the scale factor — more vertically stable
    if use_torso_height:
        scale = torso_height.replace(0, np.nan)
    else:
        scale = shoulder_width.replace(0, np.nan)
    
    # ==========================================================
    # Normalize all joint coordinates
    # ==========================================================
    # Formula:
    #
    # normalized_x = (joint_x - hip_center_x) / shoulder_width
    # normalized_y = (joint_y - hip_center_y) / shoulder_width
    #
    # Why?
    #
    # Subtracting hip center:
    #   Removes dependency on position in the image.
    #
    # Dividing by shoulder width:
    #   Removes dependency on body size and camera distance.
    #
    # Result:
    #   Skeletons become much more comparable between subjects.
    #
    # This usually improves generalization and LOSO performance.
    # ==========================================================
    
    for col in subject_transformed.columns:
        if col.endswith('_x'):
            subject_transformed[col] = (subject_transformed[col] - hip_center_x) / scale
    
        elif col.endswith('_y'):
            subject_transformed[col] = (subject_transformed[col] - hip_center_y) / scale
    # ==========================================================
    # RESULT
    # ==========================================================
    # Before:
    #
    # nose_x = 620
    # nose_y = 250
    #
    # After:
    #
    # nose_x = 0.18
    # nose_y = -1.45
    #
    # Interpretation:
    # The nose is:
    #   0.18 shoulder-widths to the right of body center
    #   1.45 shoulder-widths above body center
    #
    # These values are far more meaningful than raw pixels.
    # ==========================================================
    return subject_transformed

# English:
# Purpose: Filter out frames with abnormally small shoulder width (likely poor pose detections).
# Parameters:
# - un_norm_subject: Original DataFrame before normalization (to compute shoulder width).
# - norm_subject: Normalized DataFrame to filter.
# Returns:
# - Filtered DataFrame with only frames having shoulder_width >= SHOULDER_WIDTH_THRESHOLD.
# 日本語:
# 目的: 異常に小さい肩幅のフレーム（ポーズ検出エラー）を除外します。
# パラメータ:
# - un_norm_subject: 正規化前の元のDataFrame（肩幅計算用）。
# - norm_subject: フィルタリング対象の正規化DataFrame。
# 戻り値:
# - SHOULDER_WIDTH_THRESHOLD以上の肩幅を持つフレームのみを含むフィルタリングされたDataFrame。
def drop_bad_frames(un_norm_subject, norm_subject):
    # Get the shoulder width using not normalized df
    shoulder_width = get_shoulder_width(un_norm_subject)

    # Calculate the good frames. Here we are considering any shoulder width more than or equal to 10 units to be good
    valid_frames = shoulder_width >= SHOULDER_WIDTH_THRESHOLD

    # Creating a copy to preserve the original df
    subject_transformed = norm_subject.loc[valid_frames].copy()
    n_bad_frames = (~valid_frames).sum()
    # print(f"Total Number of Bad Frames where shoulder widht was less than 10: {n_bad_frames}")
    return subject_transformed

# English:
# Purpose: Execute the complete preprocessing pipeline for a single subject.
# The pipeline performs data cleaning, skeleton normalization, feature extraction,
# and multi-scale temporal windowing. Intermediate results can be returned for
# inspection by specifying the desired pipeline stage.
# Parameters:
# - subject_original: DataFrame containing raw pose keypoints and the Action Label column.
# - window_sizes: Tuple of (short_window, long_window) frame sizes for windowing
#                 (default: (15, 60)).
# - return_stage: Pipeline stage to return. Supported values:
#                 'normalized' - normalized skeleton coordinates.\
#                 'features'   - frame-level features before windowing.
#                 'windowed'   - final windowed features (default).
# Returns:
# - DataFrame corresponding to the selected preprocessing stage.

# 日本語:
# 目的: 単一被験者に対して完全な前処理パイプラインを実行します。
# このパイプラインでは、データクリーニング、スケルトン正規化、
# 特徴量抽出、およびマルチスケール時間窓処理を順番に実行します。
# return_stageを指定することで、途中段階の結果を取得できます。
# パラメータ:
# - subject_original: 生の姿勢キーポイントと Action Label 列を含むDataFrame。
# - window_sizes: ウィンドウ処理用の(short_window, long_window)フレームサイズのタプル
#                 （デフォルト: (15, 60)）。
# - return_stage: 返却する前処理段階。指定可能な値:
#                 'normalized' - スケルトン正規化後のデータ。
#                 'features'   - ウィンドウ処理前のフレーム単位特徴量。
#                 'windowed'   - 最終的なウィンドウ化特徴量（デフォルト）。
# 戻り値:
# - 指定した前処理段階に対応するDataFrame。
def preprocessing_pipeline(
        subject_original, 
        window_sizes=(15, 60),
        stride=10,
        return_stage='windowed',
        drop_bad_shoulder_frames=False,
        single_window_mode=False):
    
    # 1. First we will drop any NaN labelled rows
    subject_original = subject_original.dropna(subset=['Action Label'])

    # 2. One of the subject has 'Throwing' and 'Throwing Things' as separate columnd but I want it to be uniform
    subject_original['Action Label'] = subject_original['Action Label'].replace(
        'Throwing',
        'Throwing things'
    )

    # 3. We skeleton normalized the subject using hip as the origin
    subject_normalized = modify_skeleton_normalization(subject_original)
    
    if drop_bad_shoulder_frames:
        # (OPTIONAL) 4. We dropped the bad frames for too short of shoulder width 
        subject_normalized = drop_bad_frames(subject_original, subject_normalized)

    if return_stage == 'normalized':
        return subject_normalized
    
    #=========================================================
    # FEATURE ENGINEERING
    #=========================================================

    # 5. Angles (geometric, no temporal)
    subject_angles = add_angle_features(subject_normalized)

    # 6. Range of motion features (relative positions)
    subject_rom = add_range_of_motion_features(subject_normalized)

    # 7. Velocity and acceleration (temporal, frame-level)
    subject_acc = add_acceleration_features(subject_normalized)

    # 8. Join everything at frame level
    combined = subject_normalized.reset_index(drop=True).join(
        subject_angles.reset_index(drop=True)).join(
        subject_rom.reset_index(drop=True)).join(
        subject_acc.reset_index(drop=True))

    if return_stage == 'features':
        return combined
    
    # Deprecated: Single Windowing
    if single_window_mode:
        single_windowed = create_windowed_features(combined)
        print("Single windowing mode enabled. Returning single-window features.")
        return single_windowed
    
    # 9. Multi-scale windowing
    windowed = create_multiscale_windowed_features(combined, window_sizes, stride)

    # We return the processed subject
    return windowed
    
# English:
# Purpose: Apply preprocessing pipeline to all subjects in a dataset.
# Parameters:
# - all_subject: Dictionary mapping subject IDs (FILE_NAME_SUFFIX) to DataFrames.
# - window_size: Tuple of (short_window, long_window) frame sizes for windowing (default: (15, 60)).
# Returns:
# - Dictionary with same keys mapping to preprocessed windowed DataFrames.
# 日本語:
# 目的: データセット内のすべての被験者に前処理パイプラインを適用します。
# パラメータ:
# - all_subject: 被験者ID（FILE_NAME_SUFFIX）をDataFrameにマッピングする辞書。
# - window_size: ウィンドウ処理用の(short_window, long_window)フレームサイズのタプル（デフォルト: (15, 60)）。
# 戻り値:
# - 同じキーで前処理されたウィンドウ処理DataFrameにマッピングされた辞書。
def preprocess_all_subjects(all_subject, window_size=(15, 60), stride=10, return_stage="windowed", drop_bad_shoulder_frames=False, single_window_mode=False):
    all_subject_norm = {}
    for suffix in FILE_NAME_SUFFIX:
        all_subject_norm[suffix] = preprocessing_pipeline(
            subject_original=all_subject[suffix], 
            window_sizes=window_size, 
            stride=stride,
            drop_bad_shoulder_frames=drop_bad_shoulder_frames,
            single_window_mode=single_window_mode,
            return_stage=return_stage)
    return all_subject_norm

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =================================================================
# LSTM Preprocessing
# =================================================================

def create_sequence_dataset(
    df,
    sequence_length=90,
    stride=10,
    label_col='Action Label'
):
    """
    Converts a frame-level DataFrame into 3D sequences for LSTM input.

    Unlike create_multiscale_windowed_features which aggregates frames
    into statistics, this function preserves the temporal order of frames.
    Each output sequence is a slice of consecutive frames fed to the LSTM
    in the order they occurred.

    Shape of output X: (n_sequences, sequence_length, n_features)
        - n_sequences: how many valid windows were found
        - sequence_length: number of frames per sequence (e.g. 90)
        - n_features: number of feature columns (coordinates + angles + ROM + acc)

    Shape of output y: (n_sequences,)
        - one integer label per sequence (encoded from string labels)

    Parameters
    ----------
    df : pd.DataFrame
        Frame-level data from a SINGLE subject.
        Must be the 'features' stage output from preprocessing_pipeline.
        Should include coordinates, angles, ROM, and acceleration columns.
    sequence_length : int
        Number of consecutive frames per sequence. Equivalent to window
        size in your classical ML pipeline. Paper used 90 as their best.
    stride : int
        Step size between sequence start positions.
        Lower stride = more sequences but more overlap between them.
    label_col : str
        Name of the activity label column.

    Returns
    -------
    X : np.ndarray, shape (n_sequences, sequence_length, n_features)
        3D array of frame sequences ready for LSTM input.
    y : np.ndarray, shape (n_sequences,)
        Integer-encoded activity labels, one per sequence.
    label_encoder : sklearn.LabelEncoder
        Fitted encoder — needed to convert integer predictions back
        to activity name strings after inference.
    """
    feature_cols  = [c for c in df.columns if c != label_col]
    labels_array  = df[label_col].values
    feature_array = df[feature_cols].values.astype(np.float32)
    n_frames      = len(df)

    sequences = []
    seq_labels = []

    for start in range(0, n_frames - sequence_length + 1, stride):
        end           = start + sequence_length
        window_labels = labels_array[start:end]

        # Pure window check — same as classical ML pipeline
        if len(set(window_labels)) != 1:
            continue

        sequences.append(feature_array[start:end])
        seq_labels.append(window_labels[0])

    if not sequences:
        raise ValueError(
            f"No valid sequences found. "
            f"Check that sequence_length={sequence_length} is not longer "
            f"than your activity segments."
        )

    X = np.stack(sequences, axis=0)  # shape: (n_seq, seq_len, n_features)

    # Encode string labels to integers
    # LSTM needs numeric targets, not strings
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(seq_labels)

    print(f"  Sequences created : {X.shape[0]}")
    print(f"  Sequence shape    : {X.shape[1:]}  (timesteps × features)")
    print(f"  Classes encoded   : {list(label_encoder.classes_)}")

    return X, y, label_encoder


def build_loso_sequence_datasets(all_subjects_frame_level, sequence_length=90, stride=10):
    """
    Applies create_sequence_dataset to all subjects and returns a
    dictionary ready for LOSO evaluation.

    Parameters
    ----------
    all_subjects_frame_level : dict
        Keys are subject IDs, values are frame-level DataFrames
        (return_stage='features' output from preprocessing_pipeline).
    sequence_length : int
        Frames per sequence.
    stride : int
        Step between sequences.

    Returns
    -------
    dict
        Keys are subject IDs.
        Values are dicts with keys 'X', 'y', 'encoder'.
    """
    result = {}

    for subject_id, df in all_subjects_frame_level.items():
        print(f"\nSubject {subject_id}:")
        X, y, encoder = create_sequence_dataset(df, sequence_length, stride)
        result[subject_id] = {
            'X':       X,
            'y':       y,
            'encoder': encoder
        }

    return result


def scale_sequences(X_train, X_test):
    """
    Standardizes sequence data to zero mean and unit variance.
    Fitted on training data only, applied to both train and test.
    
    X shape: (n_sequences, sequence_length, n_features)
    Scaler operates on the feature dimension (axis=-1) only.
    """
    n_train, seq_len, n_features = X_train.shape
    n_test  = X_test.shape[0]

    # Reshape to 2D to fit scaler — (n_sequences * seq_len, n_features)
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d  = X_test.reshape(-1, n_features)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled  = scaler.transform(X_test_2d)

    # Reshape back to 3D
    return (
        X_train_scaled.reshape(n_train, seq_len, n_features),
        X_test_scaled.reshape(n_test,  seq_len, n_features)
    )