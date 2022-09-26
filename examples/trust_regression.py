"""
TrustReport for Regression
==========================

Simple example on how to use the TrustReport class to analyze the explanations
produced by RegressionTrustee from a MLPRegressor (neural network) from scikit-learn.
Notice that using the method `TrustReport.load()`, one can load a previously
generated report saved using `trust_report.save()`.
"""
import os

# importing required libraries
# importing Scikit-learn library and datasets package
from sklearn import datasets
from sklearn.neural_network import MLPRegressor

from trustee.report.trust import TrustReport

OUTPUT_PATH = "out/"
REPORT_PATH = f"{OUTPUT_PATH}/report/trust_report.obj"

if os.path.exists(REPORT_PATH):
    print(f"Loading trust report from {REPORT_PATH}...")
    trust_report = TrustReport.load(REPORT_PATH)
    print("Done!")
else:
    # Loading the diabetes dataset (regression)
    diabetes = datasets.load_diabetes()
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)

    # creating a RF classifier
    clf = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100, 50), max_iter=500, random_state=1)

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
        feature_names=diabetes.feature_names,
        is_classify=False,  # <----- to run the trust report for a regression model
    )

print(trust_report)
trust_report.save(OUTPUT_PATH)
