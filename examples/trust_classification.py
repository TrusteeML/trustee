"""
TrustReport for Classification
==============================

Simple example on how to use the TrustReport class to analyze the explanations
produced by ClassificationTrustee from a RandomForestClassifier from scikit-learn.
Notice that using the method `TrustReport.load()`, one can load a previously
generated report saved using `trust_report.save()`.
"""
import os

# importing required libraries
# importing Scikit-learn library and datasets package
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from trustee.report.trust import TrustReport

OUTPUT_PATH = "out/"
REPORT_PATH = f"{OUTPUT_PATH}/report/trust_report.obj"

if os.path.exists(REPORT_PATH):
    print(f"Loading trust report from {REPORT_PATH}...")
    trust_report = TrustReport.load(REPORT_PATH)
    print("Done!")
else:
    # Loading the iris plants dataset (classification)
    iris = datasets.load_iris()
    # dividing the datasets into two parts i.e. training datasets and test datasets
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)

    # creating a RF classifier
    clf = RandomForestClassifier(n_estimators=100)

    # The trust report (can) fit and explain the classifier
    trust_report = TrustReport(
        clf,
        X=X,
        y=y,
        max_iter=5,
        num_pruning_iter=5,
        train_size=0.7,
        trustee_num_iter=10,
        trustee_num_stability_iter=5,
        trustee_sample_size=0.3,
        analyze_branches=True,
        analyze_stability=True,
        top_k=10,
        verbose=True,
        class_names=iris.target_names,
        feature_names=iris.feature_names,
        is_classify=True,
    )

print(trust_report)
trust_report.save(OUTPUT_PATH)
