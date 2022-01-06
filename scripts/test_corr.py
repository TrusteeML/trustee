import matplotlib.pyplot as plt
import rootpath
import seaborn as sns
from skexplain.utils import dataset, log
from skexplain.utils.const import CIC_IDS_2017_DATASET_META

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def test_corr(dataset_meta, model=RandomForestClassifier, resampler=None, as_df=False):
    """Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model"""
    logger = log.Logger(
        "{}/res/log/test_corr_{}_{}_{}.log".format(
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y)
    logger.log("Done!")
    logger.log("#" * 10, "Done", "#" * 10)

    corr = X_test.corr()
    print(corr)
    corr.sort_values(by="Destination Port", axis=0, inplace=True)
    corr.sort_values(by="Destination Port", axis=1, inplace=True)
    plt.tight_layout()
    sns.heatmap(corr, xticklabels=1, yticklabels=1, cmap="RdYlBu")
    plt.show()

    pairs_to_drop = set()
    cols = corr.columns
    print(cols)
    for i in range(0, corr.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))

    sorted = corr.abs().unstack().drop(labels=pairs_to_drop).sort_values(ascending=False)
    print(sorted.to_string())


def main():
    """Main block"""

    # CIC_IDS_2017_DATASET_META['path'] = CIC_IDS_2017_DATASET_META['oversampled_path']
    # CIC_IDS_2017_DATASET_META['is_dir'] = False
    test_corr(CIC_IDS_2017_DATASET_META, as_df=True)


if __name__ == "__main__":
    main()
