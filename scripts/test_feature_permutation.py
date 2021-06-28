
import graphviz
import pandas as pd
import rootpath
import shap
from eli5.sklearn import PermutationImportance
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


def test_feature_permutation(dataset_meta, model=RandomForestClassifier, resampler=None, as_df=False):
    """ Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model """
    logger = log.Logger(
        "{}/res/log/test_feature_permutation_{}_{}_{}.log".format(rootpath.detect(),
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9999)
    logger.log("Done!")
    logger.log("#" * 10, "Done", "#" * 10)

    # Step 2: Train black-box model with loaded dataset
    logger.log("#" * 10, "Model train", "#" * 10)
    model_path = "../res/weights/{}_{}_{}_{}.joblib".format(model.__name__,
                                                            resampler.__name__ if resampler else "Raw",
                                                            dataset_meta['name'],
                                                            X.shape[1])
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

    perm = PermutationImportance(blackbox).fit(X_test, y_test)
    eli5.show_weights(perm)


def main():
    """ Main block """

    CIC_IDS_2017_DATASET_META['path'] = CIC_IDS_2017_DATASET_META['oversampled_path']
    CIC_IDS_2017_DATASET_META['is_dir'] = False
    test_feature_permutation(CIC_IDS_2017_DATASET_META, as_df=True)


if __name__ == "__main__":
    main()
