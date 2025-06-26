import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET_COL = 'target'

def add_time_features(df):
    """
    Получение временных компонент из transaction_time и удаление столбца
    """
    logger.info("Adding time features")

    if 'transaction_time' in df.columns:
        df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')
        dt = df['transaction_time'].dt
        df['hour'] = dt.hour
        df['year'] = dt.year
        df['month'] = dt.month
        #df['day_of_month'] = dt.day
        df['dayofweek'] = dt.dayofweek
        df.drop(columns=['transaction_time'], inplace=True)
    else:
        df['hour'] = -1
        df['year'] = -1
        df['month'] = -1
        #df['day_of_month'] = -1
        df['dayofweek'] = -1
    return df


def add_context_features(df):
    """
    Добавление контекстных фичей:
    - флаг ночи,
    - флаг выходного дня,
    - полное имя, связанное с транзакцией
    - географическая область продавца (долгота_широта)
    """
    logger.info("Adding contextual features")
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['name_full'] = df['name_1'] + '_' + df['name_2']
    df['merchant_region'] = df['merchant_lat'].round(1).astype(str) + '_' + df['merchant_lon'].round(1).astype(str)
    return df


def add_geo_features(df):
    """
    Вычисление расстояния и удаление гео столбцов
    """
    logger.info("Adding geo features")
    df['distance'] = np.sqrt((df['lat'] - df['merchant_lat'])**2 + (df['lon'] - df['merchant_lon'])**2)
    df['distance'] = np.log1p(df['distance'])
    return df


def rename_categorical(df, cat_cols):
    """
    Кодирование названий категориальных переменных
    """
    logger.debug(f"Encoding cat columns")
    rename_map = {col: f"{col}_cat" for col in cat_cols if col in df.columns}
    df.rename(columns=rename_map, inplace=True)
    return df, list(rename_map.values())


def process_numerical(train_df, input_df, num_cols, amount_median):
    """
    Заполнение пропусков в числовых фичах медианой
    и логарифмирование
    """
    logger.info(f"Processing {len(num_cols)} numerical features")

    input_df['is_large_transaction'] = (input_df['amount'] > amount_median).astype(int)

    imputer = SimpleImputer(strategy='mean')
    imputed_values = imputer.fit_transform(input_df[num_cols])
    output_df = input_df.drop(columns=num_cols)
    output_df[num_cols] = imputed_values

    for col in num_cols:
        output_df[f'{col}_log'] = np.log1p(output_df[col])
        if col != 'amount':
            output_df.drop(columns=col, inplace=True)

    return output_df


def preprocess_data(train_df, input_df):
    """
    Пайплан препроцессинга
    """
    logger.info("Starting preprocessing")

    input_df = add_time_features(input_df)
    input_df = add_context_features(input_df)
    input_df = add_geo_features(input_df)

    base_cat = [
        'merch', 'cat_id', 'name_1', 'name_2', 'gender',
        'street', 'one_city', 'us_state', 'post_code', 'jobs'
    ]
    extended_cat = base_cat + ['name_full', 'merchant_region']

    input_df, cat_cols = rename_categorical(input_df, extended_cat)

    num_cols = ['amount', 'population_city', 'distance']
    amount_median = train_df['amount'].median()
    input_df = process_numerical(train_df, input_df, num_cols, amount_median)

    logger.info(f"Preprocessing complete. Shape: {input_df.shape}")

    cat_features_idx = [input_df.columns.get_loc(col) for col in cat_cols]

    return input_df, cat_features_idx


def load_train_data(path='./train_data/train.csv'):
    logger.info('Loading training data...')

    train_df = pd.read_csv(path)
    train_df = train_df.drop(columns=['street', 'post_code'])

    logger.info('Raw train data imported. Shape: %s', train_df.shape)

    train_df = add_time_features(train_df)

    train_df = add_context_features(train_df)
    
    base_cat = ['merch', 'cat_id', 'gender', 'street', 
                'one_city', 'us_state', 'post_code', 'jobs'
    ]
    extended_cat = base_cat + ['name_full', 'merchant_region']
    train_df, _ = rename_categorical(train_df, extended_cat)

    train_df = add_geo_features(train_df)

    num_cols = ['amount', 'population_city', 'distance']
    amount_median = train_df['amount'].median()
    train_df = process_numerical(train_df, train_df, num_cols, amount_median)

    logger.info('Train data processed. Final shape: %s', train_df.shape)

    return train_df
