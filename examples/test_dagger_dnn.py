import csv

import numpy as np

import graphviz
import pandas as pd
import rootpath
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.utils import np_utils
from skexplain.imitation import ClassificationDagger, RegressionDagger
from skexplain.utils import dataset, log
from skexplain.utils.const import (
    CIC_IDS_2017_DATASET_META,
)
from sklearn import tree
from sklearn.metrics import (
    classification_report,
    f1_score,
    r2_score,
)
from sklearn.model_selection import train_test_split

RESULTS_FILE_NAME = "{}/res/results/dagger_test_dnn.csv"


def dagger_test_dnn(
    dataset_meta,
    validate_dataset_path="",
    resampler=None,
    max_leaves=None,
    ccp_alpha=0.0,
    num_samples=2000,
    as_df=False,
):
    """Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model"""
    logger = log.Logger(
        "{}/res/log/{}/dagger_test_dnn_{}_{}.log".format(
            rootpath.detect(),
            dataset_meta["name"],
            "DNN",
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

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X = scaler.fit_transform(X)
    if len(y.shape) >= 2 and not isinstance(y, pd.DataFrame):
        y = y.ravel()

    logger.log("Splitting dataset into training and test...")
    X_indexes = np.arange(0, X.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(
        X_indexes, y, train_size=0.7, stratify=y
    )
    X_train = X.iloc[X_train] if isinstance(X, pd.DataFrame) else X[X_train]
    X_test = X.iloc[X_test] if isinstance(X, pd.DataFrame) else X[X_test]
    y_train = np_utils.to_categorical(y_train)
    # y_test = y_test.argmax(axis=-1)
    logger.log("#" * 10, "Done", "#" * 10)

    # Step 2: Train black-box model with loaded dataset
    logger.log("#" * 10, "Model init", "#" * 10)
    model_path = "../res/weights/{}_{}_{}.h5".format(
        "DNN", resampler.__name__ if resampler else "Raw", dataset_meta["name"]
    )
    logger.log("Looking for pre-trained model: {}...".format(model_path))
    try:
        blackbox = load_model(model_path)
    except ValueError:
        blackbox = Sequential()
        blackbox.add(Dense(200, input_dim=X_train.shape[1], activation="relu"))
        blackbox.add(Dropout(0.5))
        blackbox.add(Dense(100, activation="softmax"))
        blackbox.add(Dropout(0.5))
        blackbox.add(Dense(30, activation="softmax"))
        blackbox.add(Dropout(0.2))
        blackbox.add(Dense(10, activation="softmax"))
        blackbox.add(Dropout(0.2))
        blackbox.add(Dense(5, activation="softmax"))
        blackbox.add(Dense(y_train.shape[1], activation="softmax"))
        logger.log(blackbox.summary())

        logger.log("#" * 10, "Model compile", "#" * 10)
        blackbox.compile(
            optimizer="adam",
            metrics=["accuracy" if dataset_meta["type"] == "classification" else "mse"],
            loss="categorical_crossentropy"
            if dataset_meta["type"] == "classification"
            else "mean_squared_error",
        )

        logger.log("#" * 10, "Model fit", "#" * 10)
        blackbox.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=100,
            verbose=2,
            batch_size=int(X_train.shape[0] / 100),
            workers=8,
        )
        blackbox.save(model_path)

    logger.log("#" * 10, "Done", "#" * 10)

    logger.log("#" * 10, "Model test", "#" * 10)
    y_pred_probs = blackbox.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=-1)

    blackbox_score = 0
    if dataset_meta["type"] == "classification":
        logger.log("Blackbox model training classification report:")
        logger.log("\n{}".format(classification_report(y_test, y_pred, digits=3)))
        blackbox_score = f1_score(y_test, y_pred, average="macro")
        logger.log(
            "F1-score for test data: {}".format(
                f1_score(y_test, y_pred, average="macro")
            )
        )
    else:
        blackbox_score = r2_score(y_test, y_pred)
        logger.log("Blackbox model R2 score: {}".format(blackbox_score))

    logger.log("#" * 10, "Done", "#" * 10)

    if dataset_meta["type"] == "classification":
        logger.log("Using Classification Dagger algorithm...")
        dagger = ClassificationDagger(expert=blackbox, logger=logger)
    else:
        logger.log("Using Regression Dagger algorithm...")
        dagger = RegressionDagger(expert=blackbox, logger=logger)

    dagger.fit(
        X,
        y,
        max_iter=100,
        max_leaf_nodes=max_leaves,
        num_samples=num_samples,
        ccp_alpha=ccp_alpha,
        verbose=True,
    )

    with open(RESULTS_FILE_NAME.format(rootpath.detect()), "a") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        logger.log("#" * 10, "Explanation validation", "#" * 10)
        (dt, reward, idx) = dagger.explain()
        logger.log("Model explanation {} local fidelity: {}".format(idx, reward))
        dt_y_pred = dt.predict(X_test)

        dt_score = 0
        fidelity = 0
        if dataset_meta["type"] == "classification":
            logger.log("Model explanation classification report:")
            logger.log(
                "\n{}".format(classification_report(y_test, dt_y_pred, digits=3))
            )
            dt_score = f1_score(y_test, dt_y_pred, average="macro")

            logger.log("Model explanation global fidelity report:")
            logger.log(
                "\n{}".format(classification_report(y_pred, dt_y_pred, digits=3))
            )
            fidelity = f1_score(y_pred, dt_y_pred, average="macro")
        else:
            dt_score = r2_score(y_test, dt_y_pred)
            fidelity = r2_score(y_pred, dt_y_pred)
            logger.log("Model explanation validation R2 score: {}".format(dt_score))
            logger.log("Model explanation global fidelity: {}".format(fidelity))

        csv_writer.writerow(
            [
                dataset_meta["name"],
                len(X),
                "DNN",
                resampler.__name__ if resampler else "None",
                dt.get_n_leaves(),
                ccp_alpha,
                blackbox_score,
                dt_score,
                fidelity,
            ]
        )

        dot_data = tree.export_graphviz(
            dt,
            feature_names=feature_names,
            class_names=dataset_meta["classes"] if "classes" in dataset_meta else None,
            filled=True,
            rounded=True,
            special_characters=True,
        )
        graph = graphviz.Source(dot_data)
        graph.render(
            "{}/res/img/{}/{}/dt_{}_{}".format(
                rootpath.detect(),
                dataset_meta["name"],
                "dagger",
                resampler.__name__ if resampler else "Raw",
                dt.get_n_leaves(),
            )
        )
        logger.log("#" * 10, "Done", "#" * 10)


def main():
    """Main block"""

    # overwrites current results to start new tests, and writes first row
    with open(RESULTS_FILE_NAME.format(rootpath.detect()), "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(
            [
                "dataset",
                "dataset size",
                "model",
                "resampler",
                "num leaves",
                "ccp_alpha",
                "blackbox f1/r2",
                "DT f1/r2",
                "fidelity",
            ]
        )

    # read already undersampled dataset from disk instead of doing the oversampling every time
    CIC_IDS_2017_DATASET_META["path"] = CIC_IDS_2017_DATASET_META["oversampled_path"]
    CIC_IDS_2017_DATASET_META["is_dir"] = False
    dagger_test_dnn(CIC_IDS_2017_DATASET_META, num_samples=100000, as_df=True)


if __name__ == "__main__":
    main()
