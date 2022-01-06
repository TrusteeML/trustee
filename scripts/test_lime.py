import io

import pandas as pd
import rootpath
from lime.lime_tabular import LimeTabularExplainer
from skexplain.utils import dataset, log, persist
from skexplain.utils.const import CIC_IDS_2017_DATASET_META
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DF_TEST = """
    Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets, Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std, Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std, Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min, Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length, Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance, FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate, Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes, Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward, Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max, Idle Min, Label\n
    444, 119302728, 2685, 1729, 8299, 7556917, 517, 0, 3.090875233, 16.85842056, 17376, 0, 4370.686524, 2566.935004, 63411.92802, 36.99831575, 27034.38205, 174625.7584, 5024984, 0, 119000000, 44449.50857, 222461.7159, 5025702, 0, 119000000, 69040.90509, 273867.424, 5024984, 1, 0, 0, 0, 0, 85928, 55336, 22.50577204, 14.49254371, 0, 17376, 1713.525708, 2669.389319, 7125639.337, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1713.91391, 3.090875233, 4370.686524, 85928, 0, 0, 0, 0, 0, 0, 2685, 8299, 1729, 7556917, 29200, 235, 118, 32, 2217, 0, 2217, 2217, 5024984, 0, 5024984, 5024984, Heartbleed\n 
"""  # noqa: E501


def lime_test(dataset_meta, model=RandomForestClassifier, resampler=None, as_df=False):
    """Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model"""
    logger = log.Logger(
        "{}/res/log/lime_test_{}_{}_{}.log".format(
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

    # Step 2: Train black-box model with loaded dataset
    logger.log("#" * 10, "Model train", "#" * 10)
    model_path = "../res/weights/{}_{}_{}.joblib".format(model.__name__, resampler.__name__ if resampler else "Raw", dataset_meta["name"])
    logger.log("Looking for pre-trained model: {}...".format(model_path))
    blackbox = persist.load_model(model_path)
    if not blackbox:
        raise ValueError("Traine model not found. Please train model before unit testing it.")

    logger.log("#" * 10, "Done", "#" * 10)

    # Step 1: Parse test def
    X_test_case, y_test_case, feature_names, _, _ = dataset.read(
        io.StringIO(DF_TEST), metadata=dataset_meta, verbose=True, logger=logger, as_df=as_df
    )

    y_pred = blackbox.predict(X_test_case)

    logger.log("Predicted class =", y_pred[0])
    logger.log("True class =", y_test_case.iloc[0][0] if isinstance(y_test_case, pd.DataFrame) else y_test_case[0][0])

    explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=dataset_meta["classes"], discretize_continuous=True)
    exp = explainer.explain_instance(X_test_case[0], blackbox.predict_proba, num_features=X_test_case.shape[1], labels=[8])
    exp.save_to_file("exp.html", show_table=True, show_all=False)
    logger.log("Explanation")
    logger.log(exp.as_map())


def main():
    """Main block"""

    CIC_IDS_2017_DATASET_META["path"] = CIC_IDS_2017_DATASET_META["undersampled_path"]
    CIC_IDS_2017_DATASET_META["is_dir"] = False
    lime_test(CIC_IDS_2017_DATASET_META, as_df=False)


if __name__ == "__main__":
    main()
