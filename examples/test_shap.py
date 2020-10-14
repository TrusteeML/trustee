import graphviz
import pandas as pd
import rootpath
import shap
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from skexplain.enums.feature_type import FeatureType
from skexplain.imitation import (ClassificationDagger,
                                 IncrementalClassificationDagger,
                                 RegressionDagger)
from skexplain.utils import dataset, log, persist
from skexplain.utils.const import (BOSTON_DATASET_META,
                                   CIC_IDS_2017_DATASET_META,
                                   DIABETES_DATASET_META, IOT_DATASET_META,
                                   WINE_DATASET_META)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC


def shap_test(dataset_meta, validate_dataset_path="", method="fit", model=RandomForestClassifier, resampler=None, as_df=False):
    """ Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model """
    logger = log.Logger(
        "{}/res/log/dagger_test_{}_{}_{}.log".format(rootpath.detect(),
                                                     model.__name__,
                                                     resampler.__name__ if resampler else "Raw",
                                                     dataset_meta['name'])
    )

    # Step 1: Load training dataset
    logger.log("#" * 10, "Dataset init", "#" * 10)
    logger.log("Reading dataset fromn CSV...")
    X, y, feature_names, _, _ = dataset.read(
        dataset_meta['path'], metadata=dataset_meta, verbose=True, logger=logger, resampler=resampler, as_df=as_df)
    logger.log("Done!")

    logger.log("Splitting dataset into training and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    logger.log("Done!")
    logger.log("#" * 10, "Done", "#" * 10)

    # Step 2: Train black-box model with loaded dataset
    logger.log("#" * 10, "Model train", "#" * 10)
    model_path = "../res/weights/{}_{}_{}.joblib".format(model.__name__,
                                                         resampler.__name__ if resampler else "Raw",
                                                         dataset_meta['name'])
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

    logger.log("#" * 10, "Model test", "#" * 10)
    y_pred = blackbox.predict(X_test)
    if dataset_meta['type'] == 'classification':
        logger.log("Blackbox model training classification report:")
        logger.log("\n{}".format(classification_report(y_test, y_pred, digits=3)))
        # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
    else:
        logger.log("Blackbox model R2 score: {}".format(r2_score(y_test, y_pred)))

    logger.log("#" * 10, "Done", "#" * 10)

    if validate_dataset_path:
        # Step 2.a (optional): Test trained model with a validation dataset
        logger.log("Reading validation dataset fromn CSV...")
        X_validate, y_validate, _, _, _ = dataset.read(
            validate_dataset_path, metadata=dataset_meta, verbose=True, logger=logger)
        logger.log("Done!")

        logger.log("#" * 10, "Model validation", "#" * 10)
        y_validation_pred = blackbox.predict(X_validate)

        if dataset_meta['type'] == 'classification':
            logger.log("Blackbox model validation classification report:")
            logger.log("\n{}".format(classification_report(y_validate, y_validation_pred, digits=3)))
            # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
        else:
            logger.log("Blackbox model validation R2 score: {}".format(r2_score(y_validate, y_validation_pred)))

        logger.log("#" * 10, "Done", "#" * 10)

    # load JS visualization code to notebook
    shap.initjs()
    # explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(blackbox)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    # shap.force_plot(explainer.expected_value, shap_values[0], X[0])

    # visualize the training set predictions
    # shap.force_plot(explainer.expected_value[0], shap_values[0], X, feature_names=[
    #                 name for (name, _, _) in dataset_meta['fields']])

    # create a dependence plot to show the effect of a single feature across the whole dataset
    # shap.dependence_plot("fixed acidity", shap_values, X)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X, feature_names=[name for (name, _, _) in dataset_meta['fields']])


def main():
    """ Main block """
    # shap_test(WINE_DATASET_META, model=RandomForestRegressor, as_df=True)

    shap_test(CIC_IDS_2017_DATASET_META, resampler=RandomUnderSampler)


if __name__ == "__main__":
    main()
