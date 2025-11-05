import pandas as pd


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['person_age'] <= 60]
    df = df[df['loan_percent_income'] <= 0.45]
    return df


def load_data(TRAIN_PATH, TEST_PATH):
    test_data = pd.read_csv(TEST_PATH)
    train_data = pd.read_csv(TRAIN_PATH)

    train_data = filter_data(train_data)

    target_data = train_data['loan_status']
    train_data = train_data.drop(columns=['loan_status'])

    idx = test_data['id']
    test_data = test_data.drop(columns=['id'])
    train_data = train_data.drop(columns=['id'])

    return test_data, train_data, target_data, idx

