import pandas as pd
import rootpath
from skexplain.utils import dataset, log
from skexplain.utils.const import CIC_IDS_2017_DATASET_META
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def model_eval(dataset_meta, model=RandomForestClassifier, resampler=None, as_df=False):
    """Evaluate generic Blackbox model"""
    logger = log.Logger(
        "{}/res/log/model_eval_{}_{}_{}.log".format(
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

    logger.log("Using 10-fold cross validation to evaluate the selected blackbox model...")
    blackbox = model()
    scores = cross_val_score(blackbox, X, y if isinstance(y, pd.DataFrame) else y.ravel(), cv=10, scoring="f1_macro", n_jobs=4)
    logger.log("Done!")

    for score in scores:
        logger.log(f"F1 Score: {score:0.2f}")

    logger.log(f"Avarage F1 Score: {scores.mean():0.2f} (+/- {scores.std() * 2:0.2f})")
    logger.log("#" * 10, "Done", "#" * 10)


def main():
    """Main block"""
    model_eval(CIC_IDS_2017_DATASET_META, model=DecisionTreeClassifier, resampler=None)


if __name__ == "__main__":
    main()
