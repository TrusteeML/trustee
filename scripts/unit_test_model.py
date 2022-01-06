import rootpath
from interpret import show
from interpret.blackbox import LimeTabular
from skexplain.utils import dataset, log, persist
from skexplain.utils.const import CIC_IDS_2017_DATASET_META, DOWNLOAD_DATASET_META
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, r2_score

DF_TEST = """
    Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets, Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std, Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std, Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min, Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length, Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance, FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate, Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes, Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward, Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max, Idle Min, Label\n
    444, 119302728, 2685, 1729, 8299, 7556917, 517, 0, 3.090875233, 16.85842056, 17376, 0, 4370.686524, 2566.935004, 63411.92802, 36.99831575, 27034.38205, 174625.7584, 5024984, 0, 119000000, 44449.50857, 222461.7159, 5025702, 0, 119000000, 69040.90509, 273867.424, 5024984, 1, 0, 0, 0, 0, 85928, 55336, 22.50577204, 14.49254371, 0, 17376, 1713.525708, 2669.389319, 7125639.337, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1713.91391, 3.090875233, 4370.686524, 85928, 0, 0, 0, 0, 0, 0, 2685, 8299, 1729, 7556917, 29200, 235, 118, 32, 2217, 0, 2217, 2217, 5024984, 0, 5024984, 5024984, Heartbleed\n
"""

DF_TEST_2 = """
    dst_port, flow_duration, tot_fwd_pkts, tot_bwd_pkts, totlen_fwd_pkts, totlen_bwd_pkts, fwd_pkt_len_max, fwd_pkt_len_min, fwd_pkt_len_mean, fwd_pkt_len_std, bwd_pkt_len_max, bwd_pkt_len_min, bwd_pkt_len_mean, bwd_pkt_len_std, flow_byts_s, flow_pkts_s, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min, fwd_iat_tot, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min, bwd_iat_tot, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min, fwd_psh_flags, bwd_psh_flags, fwd_urg_flags, bwd_urg_flags, fwd_header_len_2, bwd_header_len, fwd_pkts_s, bwd_pkts_s, pkt_len_min, pkt_len_max, pkt_len_mean, pkt_len_std, pkt_len_var, fin_flag_cnt, syn_flag_cnt, rst_flag_cnt, psh_flag_cnt, ack_flag_cnt, urg_flag_cnt, cwe_flag_cnt, ece_flag_cnt, down_up_ratio, pkt_size_avg, fwd_seg_size_avg, bwd_seg_size_avg, fwd_header_len, fwd_byts_b_avg, fwd_pkts_b_avg, fwd_blk_rate_avg, bwd_byts_b_avg, bwd_pkts_b_avg, bwd_blk_rate_avg, subflow_fwd_pkts, subflow_bwd_pkts, subflow_fwd_byts, subflow_bwd_byts, init_fwd_win_byts, init_bwd_win_byts, fwd_act_data_pkts, fwd_seg_size_min, active_mean, active_std, active_max, active_min, idle_mean, idle_std, idle_max, idle_min, label\n
    8888, 922006.0, 4026, 18618, 225756, 235402945, 344.0, 56.0, 56.07451564828614, 4.542279314120817, 16388.0, 56.0, 12643.836341175207, 6802.957047860392, 255560919.34325808, 24559.493105250942, 40.71925098264364, 173.33406314508713, 11254.0, 0.0, 922006.0, 229.06981366459627, 405.20607053461316, 11254.0, 2.0, 921840.0, 49.51603373260998, 204.4605776460379, 11561.0, 0.0, 0, 0, 0, 0, 80520, 372360, 4366.565944256328, 20192.927160994615, 56, 16388, 10405.789657304364, 7823.999035290016, 61214960.904219106, 1, 0, 0, 0, 0, 0, 0, 0, 4.624441132637854, 10405.789657304364, 56.07451564828614, 12643.836341175207, 80520, 0.0, 0.0, 0.0, 234360325.0, 18613.0, 257559480.0462015, 4026, 18618, 225756, 235402945, 65535, 6375, 1, 20, 6080.125, 2963.6504279983833, 11222.0, 42.0, 6788.125, 1967.86854473946, 11251.0, 5082.0, BENIGN\n
    1900, 3003078.0, 4, 0, 808, 0, 202.0, 202.0, 202.0, 0.0, 0.0, 0.0, 0.0, 0.0, 269.0572805634752, 1.3319667354627487, 1001026.0, 388.6755287726083, 1001499.0, 1000547.0, 3003078.0, 1001026.0, 388.6755287725127, 1001498.9999999999, 1000547.0000000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 32, 0, 1.3319667354627487, 0.0, 202, 202, 202.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 202.0, 202.0, 0.0, 32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4, 0, 808, 0, 0, 0, 4, 8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, BENIGN\n
    444, 119302728, 2685, 1729, 8299, 7556917, 517, 0, 3.090875233, 16.85842056, 17376, 0, 4370.686524, 2566.935004, 63411.92802, 36.99831575, 27034.38205, 174625.7584, 5024984, 0, 119000000, 44449.50857, 222461.7159, 5025702, 0, 119000000, 69040.90509, 273867.424, 5024984, 1, 0, 0, 0, 0, 85928, 55336, 22.50577204, 14.49254371, 0, 17376, 1713.525708, 2669.389319, 7125639.337, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1713.91391, 3.090875233, 4370.686524, 85928, 0, 0, 0, 0, 0, 0, 2685, 8299, 1729, 7556917, 29200, 235, 118, 32, 2217, 0, 2217, 2217, 5024984, 0, 5024984, 5024984, Heartbleed\n
"""


def unit_test(df_test_meta, df_train_meta, model=RandomForestClassifier, as_df=False):
    """Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model"""
    logger = log.Logger("{}/res/log/{}/unit_test_{}_{}.log".format(rootpath.detect(), df_train_meta["name"], model.__name__, "Raw"))

    # Step 1: Parse test def
    X, y, _, _, _ = dataset.read(df_train_meta["path"], metadata=df_train_meta, verbose=True, logger=logger, as_df=as_df)

    # Step 1: Parse test def
    X_test, y_test, feature_names, _, _ = dataset.read(df_test_meta["path"], metadata=df_test_meta, verbose=True, logger=logger, as_df=as_df)

    # Step 2: Train black-box model with loaded dataset
    logger.log("#" * 10, "Model init", "#" * 10)
    model_path = "../res/weights/{}_{}_{}_{}.joblib".format(model.__name__, "Raw", df_train_meta["name"], X_test.shape[1])
    logger.log("Looking for pre-trained model: {}...".format(model_path))
    blackbox = persist.load_model(model_path)
    if not blackbox:
        raise ValueError("Trained model not found. Please train model before unit testing it.")
    logger.log("#" * 10, "Done", "#" * 10)

    logger.log("#" * 10, "Model test", "#" * 10)

    # for i in range(X_test.shape[1]):
    # X_test_i = copy.deepcopy(X_test)
    # X_test_i[0][i] = -1

    y_pred = blackbox.predict(X_test)
    logger.log(y_test.ravel(), y_pred)

    blackbox_score = 0
    if df_train_meta["type"] == "classification":
        logger.log("Blackbox model training classification report:")
        logger.log("\n{}".format(classification_report(y_test, y_pred, digits=3)))
        blackbox_score = f1_score(y_test, y_pred, average="macro")
        # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
    else:
        blackbox_score = r2_score(y_test, y_pred)
        logger.log("Blackbox model R2 score: {}".format(blackbox_score))

    logger.log("#" * 10, "Done", "#" * 10)

    # Blackbox explainers need a predict function, and optionally a dataset
    lime = LimeTabular(predict_fn=blackbox.predict_proba, data=X, random_state=1)

    # Pick the instances to explain, optionally pass in labels if you have them
    lime_local = lime.explain_local(X_test[:1], y_test[:1], name="LIME")

    show(lime_local)

    # # summarize the effects of all the features
    # # shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    # # for cls in range(len(df_train_meta['classes'])):
    # shap.summary_plot(shap_values[8], X_test, plot_type='bar', feature_names=feature_names)
    #
    # logger.log("#" * 10, "Done", "#" * 10)
    #
    # shap.plots.force(8, shap_values[8])


def main():
    """Main block"""
    # read already undersampled dataset from disk instead of doing the oversampling every time
    # unit_test(df_test_meta=io.StringIO(DF_TEST_2), df_train_meta=CIC_IDS_2017_DATASET_META, model=RandomForestClassifier)

    unit_test(df_test_meta=DOWNLOAD_DATASET_META, df_train_meta=CIC_IDS_2017_DATASET_META, model=RandomForestClassifier)


if __name__ == "__main__":
    main()
