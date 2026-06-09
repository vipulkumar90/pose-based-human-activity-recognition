import pandas as pd
from har_utils.config import BASE_FILE_NAME, BASE_DIR, FILE_TYPE, FILE_NAME_SUFFIX
import random


def get_subject(suffix):
    return pd.read_csv(f"{BASE_DIR}/{BASE_FILE_NAME}_{suffix}{FILE_TYPE}", header=0, index_col=0)

def get_clean_header(title='No Title', line_len=80):
    print(f"\n{'=' * line_len}")
    print(f"{title:^{line_len}}")
    print(f"{'=' * line_len}")

def get_train_test_split(all_subject, random_state=42, show_shape=False):
    rng = random.Random(random_state)
    subject_suffix_list = list(all_subject.keys())

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

def get_X_y_split_base(train, test):
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

def get_X_y_split(all_subject, random_state=42, show_shape=False):
    train, test = get_train_test_split(all_subject, random_state)
    get_X_y_split_base(train, test)