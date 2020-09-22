import dtextract
import graphviz
import rootpath
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from skexplain.enums.feature_type import FeatureType
from skexplain.imitation import ClassificationDagger
from skexplain.utils import dataset, log, persist
from skexplain.utils.const import IOT_DATASET_META, WINE_DATASET_META
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def dtextract_test(dataset_meta, validate_dataset_path="", model=RandomForestClassifier):
    """ Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model """
    logger = log.Logger(
        "{}/res/log/dtextract_test_{}.log".format(rootpath.detect(), dataset_meta['path'].split("/")[-1])
    )

    # Step 1: Load training dataset
    logger.log("############### Dataset init ###############")
    logger.log("Reading dataset fromn CSV...")
    X, y, feature_names, num_features, cat_features = dataset.read(
        dataset_meta['path'],
        metadata=dataset_meta,
        resampler=RandomUnderSampler,
        verbose=True, logger=logger,
    )
    logger.log("Num Features: ", num_features)
    logger.log("Cat Features: ", cat_features)
    logger.log("Done!")

    logger.log("Splitting dataset into training and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    logger.log("Done!")
    logger.log("############### Done ###############")

    # Step 2: Train black-box model with loaded dataset
    logger.log("############### Model train ###############")
    model_path = "../res/weights/{}_{}.joblib".format(model.__name__, dataset_meta['path'].split("/")[-1])
    logger.log("Looking for pre-trained model: {}...".format(model_path))
    blackbox = persist.load_model(model_path)
    if not blackbox:
        logger.log("Model path does not exist.")
        logger.log("Training model: {}...".format(model))
        blackbox = model()
        blackbox.fit(X_train, y_train.ravel())
        logger.log("Done!")
        if model_path:
            persist.save_model(blackbox, model_path)
    logger.log("############### Done ###############")

    logger.log("############### Model test ###############")
    y_pred = blackbox.predict(X_test)
    logger.log("Blackbox model training classification report:")
    logger.log("\n{}".format(classification_report(y_test, y_pred, digits=3)))
    # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
    logger.log("############### Done ###############")

    if validate_dataset_path:
        # Step 2.a (optional): Test trained model with a validation dataset
        logger.log("Reading validation dataset fromn CSV...")
        X_validate, y_validate, _ = dataset.read(
            validate_dataset_path, metadata=dataset_meta, verbose=True, logger=logger)
        logger.log("Done!")

        logger.log("############### Model validation ###############")
        y_validation_pred = blackbox.predict(X_validate)
        logger.log("Blackbox model validation classification report:")
        logger.log("\n{}".format(classification_report(y_validate, y_validation_pred, digits=3)))
        logger.log("############### Done ###############")

    logger.log("Using DTExtract algorithm...")
    dt = dtextract.extract(blackbox, X, y,
                           outputPath="../res/log/dtextract_16-09-23-labeled.csv",
                           trainSize=0.7,
                           maxSize=100,
                           nPts=10000,
                           nComponents=150,
                           isClassify=dataset_meta['type'] == "classification",
                           featureNames=list(feature_names),
                           numFeaturesInds=num_features,
                           catFeaturesInds=cat_features,
                           # resampler=RandomOverSampler,
                           greedyCompare=False)

    logger.log("############### Explanation validation ###############")
    dt.plot("../res/img/{}_dtExtract".format(dataset_meta['path'].split('/')[-1]))
    logger.log("############### Done ###############")


def main():
    """ Main block """
    dtextract_test(IOT_DATASET_META)
    # dtextract_test(WINE_DATASET_META, model=RandomForestRegressor)


if __name__ == "__main__":
    main()
