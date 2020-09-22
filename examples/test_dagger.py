import graphviz
import pandas as pd
import rootpath
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


def dagger_test(dataset_meta, validate_dataset_path="", method="fit", model=RandomForestClassifier, resampler=None, num_samples=2000, as_df=False):
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

    if dataset_meta['type'] == 'classification':
        logger.log("Using Classification Dagger algorithm...")
        dagger = ClassificationDagger(expert=blackbox, logger=logger)
    else:
        logger.log("Using Regression Dagger algorithm...")
        dagger = RegressionDagger(expert=blackbox, logger=logger)

    if method == "fit":
        dagger.fit(X, y, max_iter=100, max_leaf_nodes=50, num_samples=num_samples, verbose=True)
    else:
        dagger.overfit(X, y, max_leaf_nodes=50, verbose=True)

    logger.log("#" * 10, "Explanation validation", "#" * 10)
    (dt, reward, idx) = dagger.explain()
    logger.log("Model explanation {} fidelity: {}".format(idx, reward))
    dt_y_pred = dt.predict(X_test)

    if dataset_meta['type'] == 'classification':
        logger.log("Model explanation classification report:")
        logger.log("\n{}".format(classification_report(y_test, dt_y_pred, digits=3)))
        # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
    else:
        logger.log("Model explanation validation R2 score: {}".format(r2_score(y_test, dt_y_pred)))

    if validate_dataset_path:
        logger.log("#" * 10, "Decision tree validation", "#" * 10)
        y_validation_pred = dt.predict(X_validate)

        if dataset_meta['type'] == 'classification':
            logger.log("Decision tree model validation classification report:")
            logger.log("\n{}".format(classification_report(y_validate, y_validation_pred, digits=3)))
            # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
        else:
            logger.log("Decision tree model validation R2 score: {}".format(r2_score(y_validate, y_validation_pred)))

    dot_data = tree.export_graphviz(dt,
                                    feature_names=feature_names,
                                    class_names=dataset_meta['classes'] if 'classes' in dataset_meta else None,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("{}/res/img/dt_{}_{}_{}".format(rootpath.detect(),
                                                 dataset_meta['name'],
                                                 method,
                                                 resampler.__name__ if resampler else "Raw"))
    logger.log("#" * 10, "Done", "#" * 10)


def incremental_dagger_test(dataset_meta, validate_dataset_path="", method="fit", model=RandomForestClassifier, resampler=None, num_samples=2000):
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
        dataset_meta['path'], metadata=dataset_meta, verbose=True, logger=logger, resampler=resampler)
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

    if dataset_meta['type'] == 'classification':
        logger.log("Using Classification Dagger algorithm...")
        dagger = IncrementalClassificationDagger(expert=blackbox, logger=logger)
    else:
        logger.log("Using Regression Dagger algorithm...")
        dagger = RegressionDagger(expert=blackbox, logger=logger)

    if method == "fit":
        dagger.fit(X, y, max_iter=100, num_samples=num_samples, verbose=True)
    else:
        dagger.overfit(X, y, max_leaf_nodes=50, verbose=True)

    logger.log("#" * 10, "Explanation validation", "#" * 10)
    (dt, reward, idx) = dagger.explain()
    logger.log("Model explanation {} fidelity: {}".format(idx, reward))
    dt_y_pred = dt.predict(X_test)

    if dataset_meta['type'] == 'classification':
        logger.log("Model explanation classification report:")
        logger.log("\n{}".format(classification_report(y_test, dt_y_pred, digits=3)))
        # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
    else:
        logger.log("Model explanation validation R2 score: {}".format(r2_score(y_test, dt_y_pred)))

    if validate_dataset_path:
        logger.log("#" * 10, "Decision tree validation", "#" * 10)
        y_validation_pred = dt.predict(X_validate)

        if dataset_meta['type'] == 'classification':
            logger.log("Decision tree model validation classification report:")
            logger.log("\n{}".format(classification_report(y_validate, y_validation_pred, digits=3)))
            # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
        else:
            logger.log("Decision tree model validation R2 score: {}".format(r2_score(y_validate, y_validation_pred)))

    dot_data = tree.export_graphviz(dt,
                                    feature_names=feature_names,
                                    class_names=dataset_meta['classes'] if 'classes' in dataset_meta else None,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("{}/res/img/dt_{}_{}_{}".format(rootpath.detect(),
                                                 dataset_meta['name'],
                                                 method,
                                                 resampler.__name__ if resampler else "Raw"))
    logger.log("#" * 10, "Done", "#" * 10)


def main():
    """ Main block """

    # dagger_test(IOT_DATASET_META, model=MLPClassifier, method="fit", resampler=None, num_samples=10000)
    # dagger_test(IOT_DATASET_META, model=MLPClassifier,  method="fit",
    #             resampler=RandomOverSampler, num_samples=100000)
    # dagger_test(IOT_DATASET_META, model=MLPClassifier,  method="fit",
    #             resampler=RandomUnderSampler, num_samples=1000)
    #
    # dagger_test(WINE_DATASET_META, model=MLPRegressor,  method="fit", num_samples=1000)
    # dagger_test(WINE_DATASET_META, model=MLPRegressor,
    #             method="fit", resampler=RandomOverSampler, num_samples=3000)
    # dagger_test(WINE_DATASET_META, model=MLPRegressor,
    #             method="fit", resampler=RandomUnderSampler, num_samples=10)
    #
    # dagger_test(BOSTON_DATASET_META, model=MLPRegressor,  method="fit", num_samples=500)
    #
    # dagger_test(IOT_DATASET_META, model=MLPClassifier, method="overfit", resampler=None, num_samples=100000)
    # dagger_test(IOT_DATASET_META, model=MLPClassifier,  method="overfit",
    #             resampler=RandomOverSampler, num_samples=100000)
    # dagger_test(IOT_DATASET_META, model=MLPClassifier,  method="overfit",
    #             resampler=RandomUnderSampler, num_samples=1000)
    #
    # dagger_test(WINE_DATASET_META, model=MLPRegressor,  method="overfit", num_samples=1000)
    # dagger_test(WINE_DATASET_META, model=MLPRegressor,
    #             method="overfit", resampler=RandomOverSampler, num_samples=3000)
    # dagger_test(WINE_DATASET_META, model=MLPRegressor,
    #             method="overfit", resampler=RandomUnderSampler, num_samples=10)
    #
    # dagger_test(BOSTON_DATASET_META, model=MLPRegressor,  method="overfit", num_samples=500)

    # dagger_test(DIABETES_DATASET_META, model=RandomForestClassifier, method="fit", resampler=None, num_samples=10000)

    dagger_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier,
                method="fit", resampler=None, num_samples=100000, as_df=True)


if __name__ == "__main__":
    main()
