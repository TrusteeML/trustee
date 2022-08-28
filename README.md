<img src="https://github.com/TrusteeML/trustee/blob/master/docs/_static/logo.png?raw=true" alt="Trustee" width="400"/>

This package implements the `trustee` framework to extract decision tree explanation from black-box ML models.
For more information, please visit the [documentation website](https://trusteeml.github.io).

Standard AI/ML development pipeline extended by Trustee.
<img alt="Trustee" src="https://github.com/TrusteeML/trustee/blob/master/docs/_static/flowchart.png?raw=true"  width="800">

Getting Started
---------------

This section contains basic information and instructions to get started with Trustee.

### Python Version

Trustee supports `Python >=3.7`.

### Install Trustee

Use the following command to install Trustee:

```
$ pip install trustee
```

### Sample Code

```
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from trustee import ClassificationTrustee

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

trustee = ClassificationTrustee(expert=clf)
trustee.fit(X_train, y_train, num_iter=50, num_stability_iter=10, samples_size=0.3, verbose=True)
dt = trustee.explain()
dt_y_pred = dt.predict(X_test)

print("Model explanation global fidelity report:")
print(classification_report(y_pred, dt_y_pred))
print("Model explanation score report:")
print(classification_report(y_test, dt_y_pred))
```

### Usage Examples

For simple usage examples of Trustee and TrustReport, please check the `examples/` directory.

### Other Use Cases

For other examples and use cases of how Trustee can used to scrutinize ML models, listed in the table below, please check our [Use Cases repository](https://github.com/TrusteeML/emperor).

 | Use Case           | Description                                                                                                                                                 |
 | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
 | heartbleed\_case/  | Trustee application to a Random Forest Classifier for an Intrustion Detection System, trained with CIC-IDS-2017 dataset pre-computed features.              |
 | kitsune\_case/     | Trustee application to Kitsune model for anomaly detection in network traffic, trained with features extracted from Kitsune\'s Mirai attack trace.          |
 | iot\_case/         | Trustee application to Random Forest Classifier to distguish IoT devices, trained with features extracted from the pcaps from the UNSW IoT Dataset.         |
 | moon\_star\_case/  | Trustee application to Neural Network Moon and Stars Shortcut learning toy example.                                                                         |
 | nprint\_ids\_case/ | Trustee application to the nPrintML AutoGluon Tabular Predictor for an Intrustion Detection System, also trained using pcaps from the CIC-IDS-2017 dataset. |
 | nprint\_os\_case/  | Trustee application to the nPrintML AutoGluon Tabular Predictor for OS Fingerprinting, also trained using with pcaps from the CIC-IDS-2017 dataset.         |
 | pensieve\_case/    | Trustee application to the Pensieve RL model for adaptive bit-rate prediction, and comparison to related work Metis.                                        |
 | vpn\_case/         | Trustee application the 1D-CNN trained to detect VPN traffic trained with the ISCX VPN-nonVPN dataset.                                                      |

### Supported AI/ML Libraries

 | Library      | Supported          |
 | ------------ | ------------------ |
 | scikit-learn | :white_check_mark: |
 | Keras        | :white_check_mark: |
 | Tensorflow   | :white_check_mark: |
 | PyTorch      | :white_check_mark: |
 | AutoGluon    | :white_check_mark: |

## Citing us

```
@inproceedings{Jacobs2022,
	title        = {AI/ML and Network Security: The Emperor has no Clothes},
	author       = {A. S. Jacobs and R. Beltiukov and W. Willinger and R. A. Ferreira and A. Gupta and L. Z. Granville},
	year         = 2022,
	booktitle    = {Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security},
	location     = {Los Angeles, CA, USA},
	publisher    = {Association for Computing Machinery},
	address      = {New York, NY, USA},
	series       = {CCS '22}
}
```