import io
import glob
import numpy as np
import pandas as pd

from pandas.api.types import CategoricalDtype

from trustee.enums.feature_type import FeatureType


def read(path_or_buffer, metadata={}, verbose=False, logger=None, as_df=False):
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

    if verbose:
        log("CSV dataset read:")
        log(df)
        log(df.shape)
        log("Any NAN?", df.isnull().sum().sum())
        total_usage_b = df.memory_usage(deep=True).sum()
        total_usage_mb = total_usage_b / 1024**2
        mean_usage_b = df.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024**2
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
        categorical = [[] for _ in dummy_cols]
        X = pd.get_dummies(X, columns=dummy_cols)
        for i, _ in enumerate(X.columns):
            for j, _ in enumerate(dummy_cols):
                cat_feat = dummy_cols[j]
                if str(X.columns[i]).startswith(f"{str(cat_feat)}_"):
                    categorical[j].append(i)

    if verbose:
        log("Features Shape:", X.shape)
        column_names = "".join(f"{str(i)}: {str(col)}\n" for (i, col) in zip(list(range(len(X.columns))), X.columns))
        log(f"Column names:\n{column_names}")
        log("Targets shape:", y.shape, y.columns)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(-1)

    return (
        X if as_df else np.nan_to_num(X.to_numpy()),
        y if as_df else y.to_numpy(),
        X.columns,
        numerical,
        categorical,
    )
