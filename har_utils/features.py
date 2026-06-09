import pandas as pd
import numpy as np

def compute_angle(ax, ay, bx, by, cx, cy):
    """
    Compute angle at joint B, formed by A-B-C.
    Returns angle in degrees.
    """
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by

    dot = v1x * v2x + v1y * v2y
    mag1 = np.sqrt(v1x**2 + v1y**2)
    mag2 = np.sqrt(v2x**2 + v2y**2)

    cos_angle = dot / (mag1 * mag2 + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))

def add_angle_features(df):
    """
    Computes biomechanically meaningful joint angles.
    Returns a NEW DataFrame containing ONLY the angle features + Action Label.
    Combine with your coordinate DataFrame externally to test in isolation.
    """
    result = pd.DataFrame(index=df.index)

    mid_shoulder_x = (df['left_shoulder_x'] + df['right_shoulder_x']) / 2
    mid_shoulder_y = (df['left_shoulder_y'] + df['right_shoulder_y']) / 2
    mid_hip_x = (df['left_hip_x'] + df['right_hip_x']) / 2
    mid_hip_y = (df['left_hip_y'] + df['right_hip_y']) / 2

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
    result['angle_neck_torso'] = compute_angle(
        df['nose_x'],    df['nose_y'],
        mid_shoulder_x,  mid_shoulder_y,
        mid_hip_x,       mid_hip_y
    )
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

def create_windowed_features(df, window_size=30, stride=10, label_col='Action Label'):
    """
    Converts a frame-level DataFrame into window-level features.
    
    For each window of `window_size` frames (pure label only),
    computes mean, std, min, max for every numeric feature column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Frame-level data with numeric features and a label column.
        Should be from a SINGLE subject only.
    window_size : int
        Number of frames per window.
    stride : int
        Step size between windows.
    label_col : str
        Name of the activity label column.
        
    Returns
    -------
    pd.DataFrame
        One row per valid window, with aggregated features and label.
    """
    feature_cols = [c for c in df.columns if c != label_col]
    
    records = []
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

def add_range_of_motion_features(df):
    """
    Adds frame-level features capturing relative joint positions
    and distances that help discriminate low-motion activities.

    These are designed specifically to separate:
    - Eating snacks (wrist near face, oscillating)
    - Sitting quietly (wrist low, still)
    - Using phone (wrist elevated, still)
    - Biting (wrist sustained near mouth)

    Returns a NEW DataFrame with only these features.
    Combine with coordinate + angle df before windowing.
    """
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

def add_acceleration_features(df, label_col='Action Label'):
    """
    Computes frame-level velocity and acceleration for each joint.
    
    Velocity  = position difference between consecutive frames
    Acceleration = velocity difference between consecutive frames
                 = how much the motion is changing (jerkiness)

    Biting/Head banging: rhythmic acceleration spikes
    Sitting quietly:     near-zero velocity and acceleration
    Using phone:         low velocity, very low acceleration
    Attacking:           large sudden acceleration burst

    Returns a NEW DataFrame with only these features.
    Add to frame-level df BEFORE windowing.
    """
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

def create_multiscale_windowed_features(
    df,
    window_sizes=(15, 60),
    stride=10,
    label_col='Action Label'
):
    """
    Computes windowed features at multiple time scales simultaneously.
    Each output row combines statistics from a short window AND a long
    window anchored at the same starting frame.

    Both windows start at the same position — the short one captures
    recent rapid motion, the long one captures sustained posture context.
    A row is only kept if BOTH windows contain a pure single label.

    Parameters
    ----------
    df : pd.DataFrame
        Frame-level data from a SINGLE subject.
        Should include coordinates, angles, ROM, velocity, acceleration.
    window_sizes : tuple of int
        (short_window, long_window). Default (15, 60).
    stride : int
        Step between window start positions. Same for both scales.
    label_col : str
        Name of the activity label column.

    Returns
    -------
    pd.DataFrame
        One row per valid anchor point, features from both scales combined.
    """
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