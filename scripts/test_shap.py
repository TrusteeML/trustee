import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import rootpath
import shap
from skexplain.utils import dataset, log, persist
from skexplain.utils.const import CIC_IDS_2017_DATASET_META, IOT_DATASET_META
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def shap_test(dataset_meta, model=RandomForestClassifier, resampler=None, as_df=False):
    """Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model"""
    logger = log.Logger(
        "{}/res/log/shap_test_{}_{}_{}.log".format(
            rootpath.detect(), model.__name__, resampler.__name__ if resampler else "Raw", dataset_meta["name"]
        )
    )

    # Step 1: Load training dataset
    logger.log("#" * 10, "Dataset init", "#" * 10)
    logger.log("Reading dataset fromn CSV...")
    X, y, feature_names, _, _ = dataset.read(
        dataset_meta["path"], metadata=dataset_meta, verbose=True, logger=logger, resampler=resampler, as_df=as_df
    )
    logger.log("Done!")

    logger.log("Splitting dataset into training and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9999)
    logger.log("Done!")
    logger.log("#" * 10, "Done", "#" * 10)

    logger.log("X_TEST SHAPE", X_test.shape)

    # Step 2: Train black-box model with loaded dataset
    logger.log("#" * 10, "Model train", "#" * 10)
    model_path = "../res/weights/{}_{}_{}_{}.joblib".format(
        model.__name__, resampler.__name__ if resampler else "Raw", dataset_meta["name"], X.shape[1]
    )
    logger.log("Looking for pre-trained model: {}...".format(model_path))
    blackbox = persist.load_model(model_path)
    if not blackbox:
        logger.log("Model path does not exist.")
        logger.log("Training model: {}...".format(model))
        blackbox = model()
        blackbox.fit(X_train, y_train if isinstance(y_train, pd.DataFrame) else y_train.ravel())
        logger.log("Done!")
        if model_path:
            persist.save_model(blackbox, model_path)

    logger.log("#" * 10, "Done", "#" * 10)

    logger.log("#" * 10, "Init SHAP", "#" * 10)
    logger.log("SHAP Test samples:", X_test.shape[0])
    # load JS visualization code to notebook
    shap.initjs()
    # explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(blackbox)
    shap_values = explainer.shap_values(X_test)
    logger.log("SHAP values:", np.array(shap_values).shape)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    # for cls in range(len(dataset_meta["classes"])):
    # shap.summary_plot(shap_values[cls], X_test, plot_type="bar", feature_names=feature_names)


def shap_dependence_test(dataset_meta, model=RandomForestClassifier, resampler=None, as_df=False):
    """Test SHAP dependence in Blackbox model"""
    logger = log.Logger(
        "{}/res/log/shap_test_{}_{}_{}.log".format(
            rootpath.detect(), model.__name__, resampler.__name__ if resampler else "Raw", dataset_meta["name"]
        )
    )

    # Step 1: Load training dataset
    logger.log("#" * 10, "Dataset init", "#" * 10)
    logger.log("Reading dataset fromn CSV...")
    X, y, feature_names, _, _ = dataset.read(
        dataset_meta["path"], metadata=dataset_meta, verbose=True, logger=logger, resampler=resampler, as_df=as_df
    )
    logger.log("Done!")

    logger.log("Splitting dataset into training and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9999)
    logger.log("Done!")
    logger.log("#" * 10, "Done", "#" * 10)

    logger.log("X_TEST SHAPE", X_test.shape)

    # Step 2: Train black-box model with loaded dataset
    logger.log("#" * 10, "Model train", "#" * 10)
    model_path = "../res/weights/{}_{}_{}_{}.joblib".format(
        model.__name__, resampler.__name__ if resampler else "Raw", dataset_meta["name"], X.shape[1]
    )
    logger.log("Looking for pre-trained model: {}...".format(model_path))
    blackbox = persist.load_model(model_path)
    if not blackbox:
        logger.log("Model path does not exist.")
        logger.log("Training model: {}...".format(model))
        blackbox = model()
        blackbox.fit(X_train, y_train if isinstance(y_train, pd.DataFrame) else y_train.ravel())
        logger.log("Done!")
        if model_path:
            persist.save_model(blackbox, model_path)

    logger.log("#" * 10, "Done", "#" * 10)

    logger.log("#" * 10, "Init SHAP", "#" * 10)
    logger.log("SHAP Test samples:", X_test.shape[0])
    # load JS visualization code to notebook
    shap.initjs()
    # explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(blackbox)
    shap_values = explainer.shap_values(X_test)
    logger.log("SHAP values:", np.array(shap_values).shape)

    cls = 8
    # for cls in range(len(dataset_meta['classes'])):
    for feat in range(len(feature_names)):
        shap.approximate_interactions(feat, shap_values[cls], X_test)
        for i in range(3):
            shap.dependence_plot(feat, shap_values[cls], X_test, feature_names=feature_names, show=False)
        plt.savefig("dependence_{}_{}.png".format(cls, feat), dpi=300)


def main():
    """Main block"""

    # CIC_IDS_2017_DATASET_META["path"] = CIC_IDS_2017_DATASET_META["oversampled_path"]
    # CIC_IDS_2017_DATASET_META["is_dir"] = False
    shap_test(IOT_DATASET_META, as_df=True)
    # shap_dependence_test(CIC_IDS_2017_DATASET_META, as_df=True)
    # shap_dependence_test(CIC_IDS_2017_DATASET_META, as_df=True)


if __name__ == "__main__":
    main()
