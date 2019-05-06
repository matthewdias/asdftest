import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Data:
    pass

#Loads list of coordinates from given string and swap out longitudesitudes & lattitudesitudes.
def convert_coordinates(string):
    return [(lattitudes, longitudes) for (longitudes, lattitudes) in json.loads(string)]

#Randomly truncate the end of the trip's polyline points to simulattitudese partial trips.
def random_truncate(coordinates):    
    if len(coordinates) <= 1:
        return coordinates
    n = np.random.randint(len(coordinates)-1)
    if n > 0:
        return coordinates[:-n]
    else:
        return coordinates
    
#Encode the labels for the given feature across both the train and test datasets.
def encode_feature(feature, train, test):
    encoder = LabelEncoder()
    train_values = train[feature].copy()
    test_values = test[feature].copy()
    train_values[np.isnan(train_values)] = 0
    test_values[np.isnan(test_values)] = 0
    encoder.fit(pd.concat([train_values, test_values]))
    train[feature + '_ENCODED'] = encoder.transform(train_values)
    test[feature + '_ENCODED'] = encoder.transform(test_values)
    return encoder

#Extract some features from the original columns in the given dataset.
def extract_features(df):
    df['POLYLINE'] = df['POLYLINE'].apply(convert_coordinates)
    df['START_LAT'] = df['POLYLINE'].apply(lambda x: x[0][0])
    df['START_LONG'] = df['POLYLINE'].apply(lambda x: x[0][1])
    datetime_index = pd.DatetimeIndex(df['TIMESTAMP'])
    df['QUARTER_HOUR'] = datetime_index.hour * 4 + datetime_index.minute / 15   
    df['DAY_OF_WEEK'] = datetime_index.dayofweek
    df['WEEK_OF_YEAR'] = datetime_index.weekofyear - 1
    df['DURATION'] = df['POLYLINE'].apply(lambda x: 15 * len(x))

#Remove some outliers that could otherwise undermine the training's results.
def remove_outliers(df, labels):
    indices = np.where((df.DURATION > 60) & (df.DURATION <= 2 * 3600))
    df = df.iloc[indices]
    labels = labels[indices]
    bounds = ((40.032520, -8.827892), (40.787234, -8.346299))
    indices = np.where((labels[:,0]  >= bounds[0][0]) & (labels[:,1] >= bounds[0][1]) & (labels[:,0]  <= bounds[1][0]) & (labels[:,1] <= bounds[1][1]))
    df = df.iloc[indices]
    labels = labels[indices]
    return df, labels

#Loads data from CSV files, processes and caches it in pickles for faster future loading.
def load_data():
    train_cache = 'cache/train.pickle'
    train_labels_cache = 'cache/train-labels.npy'
    validation_cache = 'cache/validation.pickle'
    validation_labels_cache = 'cache/validation-labels.npy'
    test_cache = 'cache/test.pickle'
    test_labels_cache = 'cache/test-labels.npy'
    competition_test_cache = 'cache/competition-test.pickle'
    metadata_cache = 'cache/metadata.pickle'    
    if os.path.isfile(train_cache):
        train = pd.read_pickle(train_cache)
        validation = pd.read_pickle(validation_cache)
        test = pd.read_pickle(test_cache)
        train_labels = np.load(train_labels_cache)
        validation_labels = np.load(validation_labels_cache)
        test_labels = np.load(test_labels_cache)
        competition_test = pd.read_pickle(competition_test_cache)
        with open(metadata_cache, 'rb') as handle:
            metadata = pickle.load(handle)
    else:
        datasets = []
        for kind in ['train', 'test']:
            csv_file = 'datasets/%s.csv' % kind
            df = pd.read_csv(csv_file)
            df = df[df['MISSING_DATA'] == False]
            df = df[df['POLYLINE'] != '[]']
            df.drop('MISSING_DATA', axis=1, inplace=True)
            df.drop('DAY_TYPE', axis=1, inplace=True)
            df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64[s]')
            extract_features(df)
            datasets.append(df)
        train, competition_test = datasets
        client_encoder = encode_feature('ORIGIN_CALL', train, competition_test)
        taxi_encoder = encode_feature('TAXI_ID', train, competition_test)
        stand_encoder = encode_feature('ORIGIN_STAND', train, competition_test)
        train['POLYLINE_FULL'] = train['POLYLINE'].copy()  # First, keep old version handy for future reference.
        train['POLYLINE'] = train['POLYLINE'].apply(random_truncate)  # Then truncate.
        train_labels = np.column_stack([train['POLYLINE_FULL'].apply(lambda x: x[-1][0]), train['POLYLINE_FULL'].apply(lambda x: x[-1][1])])
        train, train_labels = remove_outliers(train, train_labels)
        metadata = {'n_quarter_hours': 96, 'n_days_per_week': 7, 'n_weeks_per_year': 52, 'n_client_ids': len(client_encoder.classes_), 'n_taxi_ids': len(taxi_encoder.classes_), 'n_stand_ids': len(stand_encoder.classes_)}
        train, validation, train_labels, validation_labels = train_test_split(train, train_labels, test_size=0.02)
        validation, test, validation_labels, test_labels = train_test_split(validation, validation_labels, test_size=0.5)
        train.to_pickle(train_cache)
        validation.to_pickle(validation_cache)
        test.to_pickle(test_cache)
        np.save(train_labels_cache, train_labels)
        np.save(validation_labels_cache, validation_labels)
        np.save(test_labels_cache, test_labels)
        competition_test.to_pickle(competition_test_cache)
        with open(metadata_cache, 'wb') as handle:
            pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    data = Data()
    data.__dict__.update({'train': train, 'train_labels': train_labels, 'validation': validation, 'validation_labels': validation_labels, 'test': test, 'test_labels': test_labels, 'competition_test': competition_test, 'metadata': metadata})
    return data
