import pandas as pd
import numpy as np
import warnings


# English:
# Purpose: Compute the angle at joint B formed by joints A-B-C.
# Parameters:
# - ax, ay: Coordinates of joint A.
# - bx, by: Coordinates of joint B (the pivot point).
# - cx, cy: Coordinates of joint C.
# Returns:
# - Angle in degrees as a numpy scalar or array, clamped to [0, 180].
# 日本語:
# 目的: 関節A-B-Cで構成される関節Bの角度を計算します。
# パラメータ:
# - ax, ay: 関節Aの座標。
# - bx, by: 関節Bの座標（ピボットポイント）。
# - cx, cy: 関節Cの座標。
# 戻り値:
# - 度（°）単位の角度（[0, 180]の範囲にクリップ）。
def compute_angle(ax, ay, bx, by, cx, cy):

    # Construct vectors BA and BC from the joint coordinates
    # 関節座標からベクトルBAとBCを生成する
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by

    # Compute the dot product and vector magnitudes
    # 内積および各ベクトルの大きさを計算する
    dot = v1x * v2x + v1y * v2y
    mag1 = np.sqrt(v1x**2 + v1y**2)
    mag2 = np.sqrt(v2x**2 + v2y**2)

    # Compute the cosine of the angle and clamp it for numerical stability
    # 数値誤差を防ぐため、余弦値を計算して有効範囲に制限する
    cos_angle = dot / (mag1 * mag2 + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    
    # Convert the angle from radians to degrees
    # 角度をラジアンから度（°）へ変換する
    return np.degrees(np.arccos(cos_angle))


# English:
# Purpose: Extract upper-limb, lower-limb, and torso angle features from pose keypoints.
# Parameters:
# - df: Frame-level DataFrame with keypoint coordinates (shoulder, elbow, wrist, hip, knee, ankle, nose).
# Returns:
# - New DataFrame with only joint angle columns, same index as input.
# 日本語:
# 目的: ポーズキーポイントから上肢、下肢、胴体の角度特徴量を抽出します。
# パラメータ:
# - df: キーポイント座標を含むフレームレベルのDataFrame。
# 戻り値:
# - 関節角度列のみを含む新しいDataFrame、入力と同じインデックス。
def add_angle_features(df):
    
    result = pd.DataFrame(index=df.index)

    # Compute body reference points used for torso-related angles
    # 胴体関連の角度計算に使用する身体の基準点を計算する
    mid_shoulder_x = (df['left_shoulder_x'] + df['right_shoulder_x']) / 2
    mid_shoulder_y = (df['left_shoulder_y'] + df['right_shoulder_y']) / 2
    mid_hip_x = (df['left_hip_x'] + df['right_hip_x']) / 2
    mid_hip_y = (df['left_hip_y'] + df['right_hip_y']) / 2

    # Compute upper-limb joint angles
    # 上肢の関節角度を計算する
    result['angle_right_elbow'] = compute_angle(
        df['right_shoulder_x'], df['right_shoulder_y'],
        df['right_elbow_x'],    df['right_elbow_y'],
        df['right_wrist_x'],    df['right_wrist_y']
    )
    result['angle_left_elbow'] = compute_angle(
        df['left_shoulder_x'], df['left_shoulder_y'],
        df['left_elbow_x'],    df['left_elbow_y'],
        df['left_wrist_x'],    df['left_wrist_y']
    )
    result['angle_right_shoulder'] = compute_angle(
        df['right_hip_x'],      df['right_hip_y'],
        df['right_shoulder_x'], df['right_shoulder_y'],
        df['right_elbow_x'],    df['right_elbow_y']
    )
    result['angle_left_shoulder'] = compute_angle(
        df['left_hip_x'],      df['left_hip_y'],
        df['left_shoulder_x'], df['left_shoulder_y'],
        df['left_elbow_x'],    df['left_elbow_y']
    )

    # Compute torso posture using the nose, shoulder midpoint, and hip midpoint
    # 鼻・肩中心・腰中心を用いて胴体姿勢の角度を計算する
    result['angle_neck_torso'] = compute_angle(
        df['nose_x'],    df['nose_y'],
        mid_shoulder_x,  mid_shoulder_y,
        mid_hip_x,       mid_hip_y
    )

    # Compute lower-limb joint angles
    # 下肢の関節角度を計算する
    result['angle_right_knee'] = compute_angle(
        df['right_hip_x'],   df['right_hip_y'],
        df['right_knee_x'],  df['right_knee_y'],
        df['right_ankle_x'], df['right_ankle_y']
    )
    result['angle_left_knee'] = compute_angle(
        df['left_hip_x'],   df['left_hip_y'],
        df['left_knee_x'],  df['left_knee_y'],
        df['left_ankle_x'], df['left_ankle_y']
    )
    
    return result


# ==========================================================
# DEPRECATED / 非推奨
#
# Replaced by:
#     create_multiscale_windowed_features()
#
# This implementation is retained for reference only.
# 新しいマルチスケール実装へ移行済み。
# 参考実装として保持。
# ==========================================================

# English:
# Purpose: Convert frame-level features into fixed-size temporal windows (DEPRECATED).
# Use create_multiscale_windowed_features() instead.
# Parameters:
# - df: Frame-level DataFrame with features and action labels.
# - window_size: Number of frames per window (default: 30).
# - stride: Number of frames between window starts (default: 10).
# - label_col: Name of the activity label column (default: 'Action Label').
# Returns:
# - DataFrame with windowed features (mean, std, min, max) and activity labels.
# Side effect: Issues DeprecationWarning to console.
# 日本語:
# 目的: フレームレベル特徴量を固定長の時間窓へ変換します（非推奨）。
# create_multiscale_windowed_features()を使用してください。
# パラメータ:
# - df: 特徴量とアクションラベルを含むフレームレベルのDataFrame。
# - window_size: 1ウィンドウあたりのフレーム数（デフォルト: 30）。
# - stride: ウィンドウ開始位置間のフレーム数（デフォルト: 10）。
# - label_col: アクティビティラベル列の名前（デフォルト: 'Action Label'）。
# 戻り値:
# - ウィンドウ処理された特徴量（平均、標準偏差、最小値、最大値）とラベルのDataFrame。
# 副作用: DeprecationWarningをコンソールに出力します。
def create_windowed_features(df, window_size=30, stride=10, label_col='Action Label'):

    warnings.warn(
        "create_windowed_features() is deprecated. "
        "Use create_multiscale_windowed_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Select feature columns while excluding the activity label
    # ラベル列を除いた特徴量列を取得する
    feature_cols = [c for c in df.columns if c != label_col]
    
    records = []

    # Convert DataFrame columns to NumPy arrays for faster processing
    # 高速処理のためDataFrameをNumPy配列へ変換する
    labels_array = df[label_col].values
    feature_array = df[feature_cols].values
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        window_labels = labels_array[start:end]

        # Pure window check
        if len(set(window_labels)) != 1:
            continue

        window_data = feature_array[start:end]
        
        means = window_data.mean(axis=0)
        stds  = window_data.std(axis=0)
        mins  = window_data.min(axis=0)
        maxs  = window_data.max(axis=0)

        feature_vector = np.concatenate([means, stds, mins, maxs])
        records.append((*feature_vector, window_labels[0]))

    stat_names = (
        [f"{c}_mean" for c in feature_cols] +
        [f"{c}_std"  for c in feature_cols] +
        [f"{c}_min"  for c in feature_cols] +
        [f"{c}_max"  for c in feature_cols]
    )

    result = pd.DataFrame(records, columns=stat_names + [label_col])
    return result


# English:
# Purpose: Extract range-of-motion and relative position features for low-motion activities.
# Parameters:
# - df: Frame-level DataFrame with keypoint coordinates (wrist, shoulder, hip, nose).
# Returns:
# - New DataFrame with ROM and relative position columns only, same index as input.
# 日本語:
# 目的: 低モーション活動の判別に役立つ可動域（ROM）と相対位置特徴量を抽出します。
# パラメータ:
# - df: キーポイント座標を含むフレームレベルのDataFrame。
# 戻り値:
# - ROM および相対位置列のみを含む新しいDataFrame、入力と同じインデックス。
def add_range_of_motion_features(df):
    result = pd.DataFrame(index=df.index)

    # ── Reference points ──────────────────────────────────────────
    # We compute midpoints to get stable body reference landmarks.
    # These are more stable than individual joints because they
    # average out small detection errors on either side.

    mid_shoulder_y = (df['left_shoulder_y'] + df['right_shoulder_y']) / 2
    mid_hip_y      = (df['left_hip_y']      + df['right_hip_y'])      / 2

    # ── Wrist height relative to shoulder ─────────────────────────
    # After hip-centering, y=0 is the hip. Shoulder is above (negative y
    # in image coords, but your normalization may flip this — check the sign).
    # 
    # A positive value means wrist is BELOW the shoulder.
    # A negative value means wrist is ABOVE the shoulder (raised arm).
    #
    # Eating/phone/biting: wrist above or near shoulder → negative or near zero
    # Sitting quietly:     wrist well below shoulder    → positive

    result['right_wrist_height_rel_shoulder'] = df['right_wrist_y'] - mid_shoulder_y
    result['left_wrist_height_rel_shoulder']  = df['left_wrist_y']  - mid_shoulder_y

    # ── Wrist proximity to nose ────────────────────────────────────
    # Euclidean distance between wrist and nose.
    #
    # Eating/biting: hand near face → small distance
    # Sitting/walking: hand away from face → large distance
    # Using phone: hand near face but still → small distance, low variance
    #
    # When windowed, the MEAN captures average proximity,
    # and the STD captures whether the hand is moving toward/away from face.

    result['right_wrist_to_nose_dist'] = np.sqrt(
        (df['right_wrist_x'] - df['nose_x']) ** 2 +
        (df['right_wrist_y'] - df['nose_y']) ** 2
    )
    result['left_wrist_to_nose_dist'] = np.sqrt(
        (df['left_wrist_x'] - df['nose_x']) ** 2 +
        (df['left_wrist_y'] - df['nose_y']) ** 2
    )

    # ── Wrist height relative to hip ──────────────────────────────
    # Since hips are the origin after normalization, this is essentially
    # just the raw wrist_y value. But making it explicit is useful because:
    # 1. It's interpretable
    # 2. When windowed, min/max will capture the vertical range of motion
    #
    # Sitting quietly: wrist near or below hip level → near zero or positive
    # Eating: wrist oscillates above hip level → negative values

    result['right_wrist_height_rel_hip'] = df['right_wrist_y'] - mid_hip_y
    result['left_wrist_height_rel_hip']  = df['left_wrist_y']  - mid_hip_y

    # ── Wrist symmetry ────────────────────────────────────────────
    # Difference in height between left and right wrist.
    #
    # Using phone: typically one hand raised, one lowered → large asymmetry
    # Eating snacks: dominant hand raised → moderate asymmetry
    # Sitting quietly: both hands low → low asymmetry
    # Attacking/throwing: one arm extended → large asymmetry

    result['wrist_height_asymmetry'] = (
        df['right_wrist_y'] - df['left_wrist_y']
    )

    # Take whichever wrist is closer to the nose
    # This handles left-handed and right-handed subjects uniformly
    result['min_wrist_to_nose_dist'] = np.minimum(
        result['right_wrist_to_nose_dist'],
        result['left_wrist_to_nose_dist']
    )
    
    # Also take whichever wrist is higher (more raised)
    result['min_wrist_height_rel_shoulder'] = np.minimum(
        result['right_wrist_height_rel_shoulder'],
        result['left_wrist_height_rel_shoulder']
    )

    return result


# English:
# Purpose: Compute frame-level velocity and acceleration for all keypoint coordinates.
# Parameters:
# - df: Frame-level DataFrame with keypoint coordinates (all _x and _y columns).
# - label_col: Name of the activity label column (default: 'Action Label').
# Returns:
# - New DataFrame with velocity and acceleration columns only (_vel, _acc suffixes), same index.
# Side effect: First two rows are zeroed to handle diff() NaN values.
# 日本語:
# 目的: すべてのキーポイント座標のフレームレベル速度および加速度を計算します。
# パラメータ:
# - df: キーポイント座標を含むフレームレベルのDataFrame。
# - label_col: アクティビティラベル列の名前（デフォルト: 'Action Label'）。
# 戻り値:
# - 速度および加速度列のみを含む新しいDataFrame（_vel、_acc接尾辞）、同じインデックス。
# 副作用: diff() NaN 値を処理するため、最初の2行はゼロに設定されます。
def add_acceleration_features(df, label_col='Action Label'):
    result = pd.DataFrame(index=df.index)

    coordinate_cols = [
        col for col in df.columns
        if (col.endswith('_x') or col.endswith('_y'))
        and col != label_col
    ]

    for col in coordinate_cols:
        # Velocity: how fast the joint is moving frame to frame
        vel = df[col].diff()

        # Acceleration: how much velocity is changing frame to frame
        acc = vel.diff()

        result[f'{col}_vel'] = vel
        result[f'{col}_acc'] = acc

    # Zero out the first two rows — diff() produces NaN there
    # which would corrupt the first window of each subject.
    # Caller must reset index before passing df in.
    result.iloc[0] = 0
    result.iloc[1] = 0

    return result


# English:
# Purpose: Compute windowed features at multiple time scales simultaneously.
# Combines statistics from a short window and a long window anchored at the same starting frame.
# Parameters:
# - df: Frame-level data from a SINGLE subject (includes coordinates, angles, ROM, velocity, acceleration).
# - window_sizes: Tuple of (short_window, long_window) sizes in frames. Default: (15, 60).
# - stride: Number of frames between window start positions.
# - label_col: Name of the activity label column.
# Returns:
# - DataFrame with one row per valid anchor point, features from both scales combined.
# 日本語:
# 目的: 複数の時間スケールで同時にウィンドウ処理された特徴量を計算します。
# 同じ開始フレームに固定された短いウィンドウと長いウィンドウの統計量を結合します。
# パラメータ:
# - df: 1被験者のフレームレベルデータ（座標、角度、ROM、速度、加速度を含む）。
# - window_sizes: (short_window, long_window)のサイズ（フレーム数）のタプル。デフォルト: (15, 60)。
# - stride: ウィンドウ開始位置間のフレーム数。
# - label_col: アクティビティラベル列の名前。
# 戻り値:
# - 有効な各アンカーポイントに対して1行、両方のスケールの特徴量を結合したDataFrame。
def create_multiscale_windowed_features(
    df,
    window_sizes=(15, 60),
    stride=10,
    label_col='Action Label'
):
    short_size, long_size = window_sizes
    feature_cols = [c for c in df.columns if c != label_col]

    labels_array  = df[label_col].values
    feature_array = df[feature_cols].values
    n_frames      = len(df)

    records = []

    # We need enough frames for the LONG window
    # Short window is always contained within or equal to long window
    # Try different strides
    # Try not discarding the window where the transistioning is happeening.
    for start in range(0, n_frames - long_size + 1, stride):

        # ── Short window ──────────────────────────────────────────
        short_end    = start + short_size
        short_labels = labels_array[start:short_end]

        if len(set(short_labels)) != 1:
            continue  # short window crosses a boundary

        # ── Long window ───────────────────────────────────────────
        long_end    = start + long_size
        long_labels = labels_array[start:long_end]

        if len(set(long_labels)) != 1:
            continue  # long window crosses a boundary

        # ── Both windows must agree on the label ──────────────────
        if short_labels[0] != long_labels[0]:
            continue  # shouldn't happen since short ⊂ long, but safety check

        # ── Compute stats for each scale ──────────────────────────
        short_data = feature_array[start:short_end]
        long_data  = feature_array[start:long_end]

        def window_stats(data):
            return np.concatenate([
                data.mean(axis=0),
                data.std(axis=0),
                data.min(axis=0),
                data.max(axis=0)
            ])

        short_features = window_stats(short_data)
        long_features  = window_stats(long_data)

        # Concatenate both scales into one feature vector
        combined = np.concatenate([short_features, long_features])
        records.append((*combined, long_labels[0]))

    # ── Build column names ────────────────────────────────────────
    stat_suffixes = ['_mean', '_std', '_min', '_max']
    short_cols = [
        f'short_{c}{s}'
        for s in stat_suffixes
        for c in feature_cols
    ]
    long_cols = [
        f'long_{c}{s}'
        for s in stat_suffixes
        for c in feature_cols
    ]

    result = pd.DataFrame(
        records,
        columns=short_cols + long_cols + [label_col]
    )
    return result