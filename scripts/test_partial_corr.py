
import graphviz
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import rootpath
import seaborn as sns
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
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC


def test_partial_corr(dataset_meta, model=RandomForestClassifier, resampler=None, as_df=False):
    """ Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model """
    logger = log.Logger(
        "{}/res/log/test_partial_corr_{}_{}_{}.log".format(rootpath.detect(),
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y)
    logger.log("Done!")
    logger.log("#" * 10, "Done", "#" * 10)

    pcorr = pg.pairwise_corr(X_test, tail='two-sided')

    for index, row in pcorr.iterrows():
        pcorr = pcorr.append({
            "X": row["Y"],
            "Y": row["X"],
            "method": row["method"],
            "tail": row["tail"],
            "n": row["n"],
            "r": row["r"],
            "CI95%": row["CI95%"],
            "r2": row["r2"],
            "adj_r2": row["adj_r2"],
            "z": row["z"],
            "p-unc": row["p-unc"],
            "BF10":  row["BF10"],
            "power":  row["BF10"]},
            ignore_index=True)

    corr_matrix = pcorr.pivot_table(index='X', columns='Y', values='r')
    corr_matrix.sort_values(by='Destination Port', axis=0, inplace=True)
    corr_matrix.sort_values(by='Destination Port', axis=1, inplace=True)

    plt.tight_layout()
    sns.heatmap(corr_matrix, xticklabels=1, yticklabels=1, cmap="RdYlBu")
    plt.show()

    pairs_to_drop = set()
    cols = corr_matrix.columns
    for i in range(0, corr_matrix.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))

    sorted = corr_matrix.abs().unstack().drop(labels=pairs_to_drop).sort_values(ascending=False)
    print(sorted.to_string())


def main():
    """ Main block """

    # CIC_IDS_2017_DATASET_META['path'] = CIC_IDS_2017_DATASET_META['oversampled_path']
    # CIC_IDS_2017_DATASET_META['is_dir'] = False
    test_partial_corr(CIC_IDS_2017_DATASET_META, as_df=True)


if __name__ == "__main__":
    main()
