import csv

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, f1_score, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def model_eval(dataset_meta, model=RandomForestClassifier, resampler=None, as_df=False):
    """ Evaluate generic Blackbox model """
    logger = log.Logger(
        "{}/res/log/model_eval_{}_{}_{}.log".format(rootpath.detect(),
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

    logger.log("Using 10-fold cross validation to evaluate the selected blackbox model...")
    blackbox = model()
    scores = cross_val_score(blackbox, X, y if isinstance(y, pd.DataFrame)
                             else y.ravel(), cv=10, scoring='f1_macro', n_jobs=4)
    logger.log("Done!")

    for score in scores:
        logger.log("F1 Score: %0.2f" % (score))

    logger.log("Avarage F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    logger.log("#" * 10, "Done", "#" * 10)


def main():
    """ Main block """
    model_eval(CIC_IDS_2017_DATASET_META, model=DecisionTreeClassifier, resampler=None)


if __name__ == "__main__":
    main()
