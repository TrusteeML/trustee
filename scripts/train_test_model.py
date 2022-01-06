import numpy as np

import pandas as pd
import rootpath
from skexplain.utils import dataset, log, persist
from skexplain.utils.const import CIC_IDS_2017_DATASET_META
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, r2_score
from sklearn.model_selection import train_test_split


def train_test_model(
    dataset_meta,
    validate_dataset_path="",
    model=RandomForestClassifier,
    resampler=None,
    as_df=False,
):
    """Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model"""
    logger = log.Logger(
        "{}/res/log/{}/train_model_{}_{}.log".format(
            rootpath.detect(),
            dataset_meta["name"],
            model.__name__,
            resampler.__name__ if resampler else "Raw",
        )
    )

    # Step 1: Load training dataset
    logger.log("#" * 10, "Dataset init", "#" * 10)
    logger.log("Reading dataset fromn CSV...")
    X, y, feature_names, _, _ = dataset.read(
        dataset_meta["path"],
        metadata=dataset_meta,
        verbose=True,
        logger=logger,
        resampler=resampler,
        as_df=as_df,
    )
    logger.log("Done!")

    logger.log("Splitting dataset into training and test...")
    X_indexes = np.arange(0, X.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(X_indexes, y, train_size=0.7, stratify=y)
    X_train = X.iloc[X_train] if isinstance(X, pd.DataFrame) else X[X_train]
    X_test = X.iloc[X_test] if isinstance(X, pd.DataFrame) else X[X_test]
    logger.log("Done!")
    logger.log("#" * 10, "Done", "#" * 10)

    # Step 2: Train black-box model with loaded dataset
    logger.log("#" * 10, "Model train", "#" * 10)
    model_path = "../res/weights/{}_{}_{}_{}.joblib".format(
        model.__name__,
        resampler.__name__ if resampler else "Raw",
        dataset_meta["name"],
        X.shape[1],
    )
    logger.log("Training model: {}...".format(model))

    logger.log("y_train", y_train)
    try:
        blackbox = model(n_jobs=4)
    except Exception:
        blackbox = model()
    blackbox.fit(X_train, y_train if isinstance(y_train, pd.DataFrame) else y_train.ravel())
    logger.log("Done!")
    if model_path:
        persist.save_model(blackbox, model_path)

    logger.log("#" * 10, "Done", "#" * 10)

    logger.log("#" * 10, "Model test", "#" * 10)
    y_pred = blackbox.predict(X_test)

    blackbox_score = 0
    if dataset_meta["type"] == "classification":
        logger.log("Blackbox model training classification report:")
        logger.log("\n{}".format(classification_report(y_test, y_pred, digits=3)))
        blackbox_score = f1_score(y_test, y_pred, average="macro")
        # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
    else:
        blackbox_score = r2_score(y_test, y_pred)
        logger.log("Blackbox model R2 score: {}".format(blackbox_score))

    logger.log("#" * 10, "Done", "#" * 10)

    if validate_dataset_path:
        # Step 2.a (optional): Test trained model with a validation dataset
        logger.log("Reading validation dataset fromn CSV...")
        X_validate, y_validate, _, _, _ = dataset.read(validate_dataset_path, metadata=dataset_meta, verbose=True, logger=logger)
        logger.log("Done!")

        logger.log("#" * 10, "Model validation", "#" * 10)
        y_validation_pred = blackbox.predict(X_validate)

        if dataset_meta["type"] == "classification":
            logger.log("Blackbox model validation classification report:")
            logger.log("\n{}".format(classification_report(y_validate, y_validation_pred, digits=3)))
            # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
        else:
            logger.log("Blackbox model validation R2 score: {}".format(r2_score(y_validate, y_validation_pred)))

        logger.log("#" * 10, "Done", "#" * 10)


def main():
    """Main block"""
    # read already undersampled dataset from disk instead of doing the oversampling every time
    # CIC_ALT_DATASET_META['path'] = CIC_ALT_DATASET_META['oversampled_path']
    # CIC_ALT_DATASET_META['is_dir'] = False
    train_test_model(
        CIC_IDS_2017_DATASET_META,
        model=RandomForestClassifier,
        as_df=True,
    )


if __name__ == "__main__":
    main()
