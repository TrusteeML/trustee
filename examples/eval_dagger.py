import csv

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
from sklearn.metrics import classification_report, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC


def dagger_leaves_eval(dataset_meta, model=RandomForestClassifier, resampler=None, num_samples=2000, as_df=False):
    """ Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model """
    logger = log.Logger(
        "{}/res/log/dagger_leaves_eval_{}_{}_{}.log".format(rootpath.detect(),
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
    else:
        logger.log("Blackbox model R2 score: {}".format(r2_score(y_test, y_pred)))

    logger.log("#" * 10, "Done", "#" * 10)

    with open("{}/res/results/dagger_leaves_eval.csv".format(rootpath.detect()), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['dataset', 'method', 'model', 'resampler', 'num_leaves', 'f1/r2', 'fidelity'])
        for num_leaves in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            if dataset_meta['type'] == 'classification':
                logger.log("Using Classification Dagger algorithm...")
                dagger = ClassificationDagger(expert=blackbox, logger=logger)
            else:
                logger.log("Using Regression Dagger algorithm...")
                dagger = RegressionDagger(expert=blackbox, logger=logger)

            dagger.fit(X, y, max_iter=100, max_leaf_nodes=num_leaves, num_samples=num_samples, verbose=True)

            logger.log("#" * 10, "Explanation validation", "#" * 10)
            (dt, reward, idx) = dagger.explain()
            logger.log("Model explanation {} local fidelity: {}".format(idx, reward))
            dt_y_pred = dt.predict(X_test)

            score = 0
            if dataset_meta['type'] == 'classification':
                logger.log("Model explanation classification report:")
                logger.log("\n{}".format(classification_report(y_test, dt_y_pred, digits=3)))
                logger.log("Model explanation global fidelity report:")
                logger.log("\n{}".format(classification_report(y_pred, dt_y_pred, digits=3)))
                score = f1_score(y_test, dt_y_pred, average="macro")
                fidelity = f1_score(y_pred, dt_y_pred, average="macro")
            else:
                score = r2_score(y_test, dt_y_pred)
                fidelity = r2_score(y_pred, dt_y_pred)
                logger.log("Model explanation validation R2 score: {}".format(score))
                logger.log("Model explanation global fidelity: {}".format(fidelity))

            writer.writerow([dataset_meta['name'], 'fit', model.__name__,
                             resampler.__name__ if resampler else "Raw", num_leaves, score, fidelity])

            dot_data = tree.export_graphviz(dt,
                                            feature_names=feature_names,
                                            class_names=dataset_meta['classes'] if 'classes' in dataset_meta else None,
                                            filled=True,
                                            rounded=True,
                                            special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render("{}/res/img/leaves_eval/{}/dt_{}_{}_{}".format(rootpath.detect(),
                                                                        "dagger",
                                                                        num_leaves,
                                                                        dataset_meta['name'],
                                                                        resampler.__name__ if resampler else "Raw"))
            logger.log("#" * 10, "Done", "#" * 10)


def main():
    """ Main block """
    dagger_leaves_eval(IOT_DATASET_META, model=MLPClassifier, resampler=None, num_samples=10000)


if __name__ == "__main__":
    main()
