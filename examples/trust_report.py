import numpy as np

import pandas as pd
import rootpath

from skexplain.report import trust_report
from skexplain.utils import dataset, log, persist
from skexplain.utils.const import IOT_DATASET_META  # , CIC_IDS_2017_DATASET_META

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def trust_report_test(dataset_meta, model=RandomForestClassifier):
    """Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model"""
    logger = log.Logger(
        "{}/res/log/{}/trust_report_{}_{}.log".format(
            rootpath.detect(),
            dataset_meta["name"],
            model.__name__,
            "Raw",
        )
    )

    # Step 1: Load training dataset
    logger.log("#" * 10, "Dataset init", "#" * 10)
    logger.log("Reading dataset from CSV...")
    X, y, feature_names, _, _ = dataset.read(
        dataset_meta["path"],
        metadata=dataset_meta,
        verbose=False,
        logger=logger,
        as_df=True,
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
        "Raw",
        dataset_meta["name"],
        X.shape[1],
    )
    logger.log(f"Looking for pre-trained model: {model_path}...")
    blackbox = persist.load_model(model_path)
    if not blackbox:
        logger.log("Model path does not exist.")
        logger.log(f"Training model: {model}...")

        try:
            blackbox = model(n_jobs=4)
        except ValueError:
            blackbox = model()
        blackbox.fit(X_train, y_train if isinstance(y_train, pd.DataFrame) else y_train.ravel())
        logger.log("Done!")
        if model_path:
            persist.save_model(blackbox, model_path)
    logger.log("#" * 10, "Done", "#" * 10)

    logger.log(
        trust_report(
            blackbox,
            X=X,
            y=y,
            logger=logger,
            max_iter=1,
            dagger_num_iter=1,
            feature_names=feature_names,
            class_names=dataset_meta["classes"] if "classes" in dataset_meta else None,
            output_dir=f"{rootpath.detect()}/res/trust_report",
        )
    )


def main():
    """Main block"""
    # read already oversampling dataset from disk instead of doing the oversampling every time
    # CIC_IDS_2017_DATASET_META["path"] = CIC_IDS_2017_DATASET_META["oversampled_path"]
    # CIC_IDS_2017_DATASET_META["is_dir"] = False
    # trust_report_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier)
    trust_report_test(IOT_DATASET_META, model=RandomForestClassifier)


if __name__ == "__main__":
    main()
