import glob
import io

import numpy as np

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pandas.api.types import CategoricalDtype
from skexplain.enums.feature_type import FeatureType
from skexplain.utils.const import CIC_IDS_2017_DATASET_META


def read(path_or_buffer, metadata={}, verbose=False, logger=None, as_df=False, resampler=None):
    """
    Reads dataset from a CSV and returns it as dataframe

    Args:
        - path
            Path to read dataset from
    Returns: None
        - X
            Features from dataset as an np.array or dataframe (if as_df == True)
        - y
            Target variable from dataset as an np.array or dataframe (if as_df == True)
        - feature_names
            Feature names, if any were extracted from csv or metadata
    """

    log = logger.log if logger else print

    X = []
    y = []

    idx_offset = 0  # cant use enumerate because of skipping ID features
    names = []
    dtypes = {}
    numerical = []
    categorical = []
    dummies = []
    use_cols = []
    result = []

    for idx, (name, type, dtype, is_result) in enumerate(metadata["fields"]):
        if type != FeatureType.IDENTIFIER:
            use_cols.append(idx)

            if dtype:
                dtypes[name] = dtype

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

    if "is_dir" in metadata and metadata["is_dir"] and not isinstance(path_or_buffer, io.TextIOBase):
        df_list = []

        datasets = glob.glob(path_or_buffer + "/*.csv")
        if not datasets:
            datasets = glob.glob(path_or_buffer + "/*.zip")

        for dataset_path in datasets:
            df = pd.read_csv(
                dataset_path,
                delimiter=metadata["delimiter"] if "delimiter" in metadata else ",",
                header=0 if "has_header" in metadata and metadata["has_header"] else None,
                names=names if names else None,
                dtype=dtypes if dtypes else None,
                usecols=use_cols,
                converters=metadata["converters"] if "converters" in metadata else None,
            ).fillna(-1)
            df_list.append(df)
        df = pd.concat(df_list, axis=0, ignore_index=True)
    else:
        df = pd.read_csv(
            path_or_buffer,
            delimiter=metadata["delimiter"] if "delimiter" in metadata else ",",
            header=0 if "has_header" in metadata and metadata["has_header"] else None,
            names=names if names else None,
            dtype=dtypes if dtypes else None,
            usecols=use_cols,
            converters=metadata["converters"] if "converters" in metadata else None,
        ).fillna(-1)

    names = df.columns  # guaranteing names will be filled, even if read from csv
    if verbose:
        log("Pandas read_csv complete.")

    if "categories" in metadata:
        for (column, categories) in metadata["categories"].items():
            category = CategoricalDtype(categories=categories, ordered=True)
            df[column] = df[column].astype(category)

    # # filter only benign and specific attack(e.g slowloris == 8)
    # mask = (df[names[result]] == 1) | (df[names[result]] == 0)
    # pos = np.flatnonzero(mask)
    # df = df.iloc[pos]

    if verbose:
        log("CSV dataset read:")
        log(df)
        log(df.shape)
        log("Any NAN?", df.isnull().sum().sum())
        total_usage_b = df.memory_usage(deep=True).sum()
        total_usage_mb = total_usage_b / 1024 ** 2
        mean_usage_b = df.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        log(f"Total memory usage: {total_usage_mb:03.2f} MB")
        log(f"Average memory usage: {mean_usage_mb:03.2f} MB")

    # if no result varibles is passed in metadata, we assume the last column as the result
    if not result:
        result = [len(df.columns) - 1]

    y = df[names[result]].copy()

    X = df.drop(columns=names[result], axis=1)
    # resulting dataset corresponds to feature variables only, so encode it if necessary
    if dummies:
        dummy_cols = [names[i] for i in dummies]
        categorical = [[] for i in dummy_cols]
        X = pd.get_dummies(X, columns=dummy_cols)
        for i in range(len(X.columns)):
            for j in range(len(dummy_cols)):
                cat_feat = dummy_cols[j]
                if str(X.columns[i]).startswith(f"{str(cat_feat)}_"):
                    categorical[j].append(i)

    if verbose:
        log("Features Shape:", X.shape)
        log(
            "Column names:\n{}".format(
                "".join(
                    ("{}: {}\n".format(str(i), str(col)) for (i, col) in zip(list(range(len(X.columns))), X.columns))
                )
            )
        )

        log("Targets shape:", y.shape, y.columns)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(-1)

    if resampler:
        log(f"Resampling dataset using: {resampler.__name__}")
        try:
            if categorical:
                resample = resampler(
                    categorical_features=[item for sublist in categorical for item in sublist],
                    n_jobs=4,
                )
            else:
                resample = resampler(n_jobs=4)
        except Exception:
            resample = resampler()
        X, y = resample.fit_resample(X, y)
        log("Features Shape after resample:", X.shape)
        log("Targets shape after resample::", y.shape, y.columns)

    # if (X < 0).values.any():
    #     log("NEGATIVE VALUES DETECTED")
    #     X[X < 0] = 0
    #     if (X < 0).values.any():
    #         log("NEGATIVE VALUES STILL DETECTED")

    return (
        X if as_df else np.nan_to_num(X.to_numpy()),
        y if as_df else y.to_numpy(),
        X.columns,
        numerical,
        categorical,
    )


def read_all_categories(datasets_dir, metadata={}):
    names = []
    numerical = []
    dummies = []
    use_cols = []
    result = []

    for idx, (name, type, is_result) in enumerate(metadata["fields"]):
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
        df = pd.read_csv(
            path,
            delimiter=metadata["delimiter"] if "delimiter" in metadata else ",",
            header=0 if "has_header" in metadata and metadata["has_header"] else None,
            names=names if names else None,
            usecols=use_cols,
            converters=metadata["converters"] if "converters" in metadata else None,
        ).fillna(-1)
        dataset_list.append(df)

    frame = pd.concat(dataset_list, axis=0, ignore_index=True)
    # y = frame[names[result]].copy()
    # X = frame.drop(columns=names[result], axis=1)
    # resulting dataset corresponds to feature variables only, so encode it if necessary
    if dummies:
        dummy_cols = [names[i] for i in dummies]
        frame = pd.get_dummies(frame, columns=dummy_cols)

    print(frame.columns)


def resample(path, output_path, metadata, resampler):
    X = []
    y = []

    names = []
    result = []
    for idx, (name, type, is_result) in enumerate(metadata["fields"]):
        if name:
            names.append(name)

        if is_result:
            result.append(idx)

    print("Reading CSV dataset:", path)
    if "is_dir" in metadata and metadata["is_dir"]:
        df_list = []
        datasets = glob.glob(path + "/*.csv")
        for dataset_path in datasets:
            df = pd.read_csv(
                dataset_path,
                delimiter=metadata["delimiter"] if "delimiter" in metadata else ",",
                header=0 if "has_header" in metadata and metadata["has_header"] else None,
                names=names if names else None,
            )
            df_list.append(df)
        df = pd.concat(df_list, axis=0, ignore_index=True)
    else:
        df = pd.read_csv(
            path,
            delimiter=metadata["delimiter"] if "delimiter" in metadata else ",",
            header=0 if "has_header" in metadata and metadata["has_header"] else None,
            names=names if names else None,
        )

    names = df.columns  # guaranteing names will be filled, even if read from csv

    print("CSV dataset read:")
    print(df)

    # if no result varibles is passed in metadata, we assume the last column as the result
    if not result:
        result = [len(df.columns) - 1]

    y = df[names[result]].copy()
    X = df.drop(columns=names[result], axis=1)

    print("Features Shape:", X.shape)
    print(
        "Column names:\n{}".format(
            "".join(("{}: {}\n".format(str(i), str(col)) for (i, col) in zip(list(range(len(X.columns))), X.columns)))
        )
    )

    print("Targets shape:", y.shape, y.columns)

    print(f"Resampling dataset using: {resampler.__name__}")
    try:
        resample = resampler(n_jobs=4)
    except Exception:
        resample = resampler()

    X, y = resample.fit_resample(X, y)

    print("Features Shape after resample:", X.shape)
    print("Targets shape after resample::", y.shape, y.columns)

    print("Saving CSV dataset:", output_path)

    result_df = pd.concat([X, y], axis=1)
    result_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # read_all_categories("{}/res/dataset/CIC-IDS-2017/MachineLearningCVE/".format(rootpath.detect()),
    #                     CIC_IDS_2017_DATASET_META)

    resample(
        CIC_IDS_2017_DATASET_META["path"],
        CIC_IDS_2017_DATASET_META["oversampled_path"],
        CIC_IDS_2017_DATASET_META,
        resampler=RandomOverSampler,
    )

    resample(
        CIC_IDS_2017_DATASET_META["path"],
        CIC_IDS_2017_DATASET_META["undersampled_path"],
        CIC_IDS_2017_DATASET_META,
        resampler=RandomUnderSampler,
    )
