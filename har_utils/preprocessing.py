import numpy as np
from har_utils.config import SHOULDER_WIDTH_THRESHOLD, FILE_NAME_SUFFIX
from har_utils.features import (
    add_angle_features,
    add_range_of_motion_features,
    add_acceleration_features,
    create_multiscale_windowed_features
)

def do_sanity_check(subject_original, subject_name="Name not provided"):
    print(f"\n{'_'*60}\n")
    print(f"{' '*15}Sanity Check - {subject_name}")
    print(f"{'_'*60}")

    # NaNs per column
    n_nan = subject_original.isna().sum().to_dict()

    # Infs per numeric column
    n_inf = {
        col: np.isinf(subject_original[col]).sum()
        for col in subject_original.select_dtypes(include=np.number).columns
    }

    print("\nNaN counts:")
    for col, count in n_nan.items():
        if count:
            print(f"  {col}: {count}")

    print("\nInf counts:")
    for col, count in n_inf.items():
        if count:
            print(f"  {col}: {count}")

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
def get_shoulder_width(subject):
    shoulder_width = np.sqrt(
        (subject['left_shoulder_x'] - subject['right_shoulder_x']) ** 2 +
        (subject['left_shoulder_y'] - subject['right_shoulder_y']) ** 2
    )
    
    return shoulder_width

# =========================================================
def get_torso_height(subject):
    """
    Vertical distance from mid-hip to mid-shoulder.
    More stable vertical reference than shoulder width.
    """
    mid_shoulder_y = (subject['left_shoulder_y'] + subject['right_shoulder_y']) / 2
    mid_hip_y      = (subject['left_hip_y']      + subject['right_hip_y'])      / 2

    torso_height = np.sqrt(
        ((subject['left_shoulder_x'] + subject['right_shoulder_x']) / 2 -
         (subject['left_hip_x']      + subject['right_hip_x'])      / 2) ** 2 +
        (mid_shoulder_y - mid_hip_y) ** 2
    )
    return torso_height

# Skeleton Normalization
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

def preprocessing_pipeline(subject_original, window_sizes=(15, 60)):

    # 1. First we will drop any NaN labelled rows
    subject_original = subject_original.dropna(subset=['Action Label'])

    # 2. We skeleton normalized the subject using hip as the origin
    subject_normalized = modify_skeleton_normalization(subject_original)

    # 3. We dropped the bad frames for too short of shoulder width 
    subject_dropped_bad_frames = drop_bad_frames(subject_original, subject_normalized)

    # 4. One of the subject has 'Throwing' and 'Throwing Things' as separate columnd but I want it to be uniform
    subject_dropped_bad_frames['Action Label'] = subject_dropped_bad_frames['Action Label'].replace(
        'Throwing',
        'Throwing things'
    )

    # 5. Angles (geometric, no temporal)
    subject_angles = add_angle_features(subject_dropped_bad_frames)
    
    # 6. Range of motion features (relative positions)
    subject_rom = add_range_of_motion_features(subject_dropped_bad_frames)
    
    # 7. Velocity and acceleration (temporal, frame-level)
    subject_acc = add_acceleration_features(subject_dropped_bad_frames)
    
    # 8. Join everything at frame level
    combined = subject_dropped_bad_frames.reset_index(drop=True).join(
        subject_angles.reset_index(drop=True)).join(
        subject_rom.reset_index(drop=True)).join(
        subject_acc.reset_index(drop=True))
    
    # 9. Multi-scale windowing
    windowed = create_multiscale_windowed_features(combined, window_sizes)
    
    # We return the processed subject
    return windowed
    
def preprocess_all_subjects(all_subject, window_size=(15, 60)):
    all_subject_norm = {}
    for suffix in FILE_NAME_SUFFIX:
        all_subject_norm[suffix] = preprocessing_pipeline(all_subject[suffix], window_size)
    return all_subject_norm