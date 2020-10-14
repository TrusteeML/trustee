import glob

import numpy as np

import pandas
import rootpath
from pandas.api.types import CategoricalDtype
from skexplain.enums.feature_type import FeatureType
from skexplain.utils.const import CIC_IDS_2017_DATASET_META, IOT_DATASET_META


def read(path, metadata={}, verbose=False, logger=None, as_df=False, resampler=None):
    """
        Reads dataset from a CSV and returns it as dataframe

        Args:
            - path
                Path to read dataset from
        Returns: None
            - X
                Features from dataset as an np.array
            - y
                Target variable from dataset as an np.array
            - feature_names
                Feature names, if any were extracted from csv or metadata
    """

    log = logger.log if logger else print

    X = []
    y = []

    idx_offset = 0  # cant use enumerate because of skipping ID features
    names = []
    numerical = []
    categorical = []
    dummies = []
    use_cols = []
    result = []

    for idx, (name, type, is_result) in enumerate(metadata['fields']):
        if (type != FeatureType.IDENTIFIER):
            use_cols.append(idx)

            if name:
                names.append(name)

            if type == FeatureType.NUMERICAL and not is_result:
                numerical.append(idx - idx_offset)

            if type == FeatureType.CATEGORICAL and not is_result:
                dummies.append(idx - idx_offset)

            if is_result:
                result.append(idx - idx_offset)
        else:
            idx_offset += 1

    if verbose:
        log(10 * "=", "Metadata start.", 10 * "=")
        log("Names:", names)
        log("Dummies:", dummies)
        log("Use cols:", use_cols)
        log("Result variable:", result)
        log(10 * "=", "Metadata end.", 10 * "=")
        log("")  # skip line

    if 'is_dir' in metadata and metadata['is_dir']:
        df_list = []
        datasets = glob.glob(path + "/*.csv")
        for dataset_path in datasets:
            df = pandas.read_csv(dataset_path,
                                 delimiter=metadata['delimiter'] if 'delimiter' in metadata else ",",
                                 header=0 if 'has_header' in metadata and metadata['has_header'] else None,
                                 names=names if names else None,
                                 usecols=use_cols,
                                 converters=metadata['converters'] if 'converters' in metadata else None,
                                 verbose=verbose).fillna(-1)
            df_list.append(df)
        df = pandas.concat(df_list, axis=0, ignore_index=True)
    else:
        df = pandas.read_csv(path,
                             delimiter=metadata['delimiter'] if 'delimiter' in metadata else ",",
                             header=0 if 'has_header' in metadata and metadata['has_header'] else None,
                             names=names if names else None,
                             usecols=use_cols,
                             converters=metadata['converters'] if 'converters' in metadata else None,
                             verbose=verbose).fillna(-1)

    names = df.columns  # guaranteing names will be filled, even if read from csv

    # for i in numerical:
    #     df[names[i]].fillna(df[names[i]].mean(), inplace=True)

    if 'categories' in metadata:
        for (column, categories) in metadata['categories'].items():
            log("Column/Categories", column, categories)
            category = CategoricalDtype(categories=categories, ordered=True)
            df[column] = df[column].astype(category)

    if verbose:
        log("CSV dataset read:", df)

    # if no result varibles is passed in metadata, we assume the last column as the result
    if not result:
        result = [len(df.columns) - 1]

    y = df[names[result]].copy()
    X = df.drop(columns=names[result], axis=1)
    # resulting dataset corresponds to feature variables only, so encode it if necessary
    if dummies:
        dummy_cols = [names[i] for i in dummies]
        categorical = [[] for i in dummy_cols]
        X = pandas.get_dummies(X, columns=dummy_cols)
        for i in range(len(X.columns)):
            for j in range(len(dummy_cols)):
                cat_feat = dummy_cols[j]
                if str(X.columns[i]).startswith("{}_".format(str(cat_feat))):
                    categorical[j].append(i)

    if verbose:
        log("Features Shape:", X.shape)
        log('Column names:\n{}'.format(''.join(('{}: {}\n'.format(str(i), str(col)) for (i, col)
                                                in zip(list(range(len(X.columns))), X.columns)))))

        log("Targets shape:", y.shape, y.columns)

    if resampler:
        log("Resampling dataset using: {}".format(resampler.__name__))
        ros = resampler()
        X, y = ros.fit_resample(X, y)
        log("Features Shape after resample:", X.shape)
        log("Targets shape after resample::", y.shape, y.columns)

    for col in X.columns:
        if X[col].dtype == np.float64:
            X[col] = X[col].astype(np.float32)
        elif X[col].dtype == np.int64:
            X[col] = X[col].astype(np.int32)

    for col in y.columns:
        if y[col].dtype == np.float64:
            y[col] = y[col].astype(np.float32)
        elif y[col].dtype == np.int64:
            y[col] = y[col].astype(np.int32)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(-1)

    # if (X < 0).values.any():
    #     log("NEGATIVE VALUES DETECTED")
    #     X[X < 0] = 0
    #     if (X < 0).values.any():
    #         log("NEGATIVE VALUES STILL DETECTED")

    return X if as_df else np.nan_to_num(X.to_numpy()), y if as_df else y.to_numpy(), X.columns, numerical, categorical


def read_all_categories(datasets_dir, metadata={}):
    names = []
    numerical = []
    categorical = []
    dummies = []
    use_cols = []
    result = []

    for idx, (name, type, is_result) in enumerate(metadata['fields']):
        if name:
            names.append(name)

        if type != FeatureType.IDENTIFIER:
            use_cols.append(idx)

        if type == FeatureType.NUMERICAL:
            numerical.append(idx)

        if type == FeatureType.CATEGORICAL and not is_result:
            dummies.append(idx)

        if is_result:
            result.append(idx)

    datasets = glob.glob(datasets_dir + "/*.csv")
    dataset_list = []

    for path in datasets:
        df = pandas.read_csv(path,
                             delimiter=metadata['delimiter'] if 'delimiter' in metadata else ",",
                             header=0 if 'has_header' in metadata and metadata['has_header'] else None,
                             names=names if names else None,
                             usecols=use_cols,
                             converters=metadata['converters'] if 'converters' in metadata else None).fillna(-1)
        dataset_list.append(df)

    frame = pandas.concat(dataset_list, axis=0, ignore_index=True)
    # y = frame[names[result]].copy()
    # X = frame.drop(columns=names[result], axis=1)
    # resulting dataset corresponds to feature variables only, so encode it if necessary
    if dummies:
        dummy_cols = [names[i] for i in dummies]
        categorical = [[] for i in dummy_cols]
        frame = pandas.get_dummies(frame, columns=dummy_cols)

    print(frame.columns)


if __name__ == "__main__":
    read_all_categories("{}/res/dataset/CIC-IDS-2017/MachineLearningCVE/".format(rootpath.detect()),
                        CIC_IDS_2017_DATASET_META)
