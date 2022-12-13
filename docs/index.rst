.. image:: _static/logo.png
  :width: 400px
  :align: center
  :alt: Trustee
  :class: only-light

.. image:: _static/logo-alt.png
  :width: 400px
  :align: center
  :alt: Trustee
  :class: only-dark

Welcome to Trustee's documentation. Get started with `installation`
and then get an overview with the `quickstart`. The rest of the docs
describe each component of Trustee in detail, with a full reference in 
the :doc:`api` section.

.. raw:: html

  <style>
    .buttons {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end; 
    }

    .buttons a {
      display: flex;
      align-items: center;
      margin: 5px;
    }

    .buttons svg {
      font-size: 1.5em;
      height: 1em;
      width: 1em;
    }

    .buttons .label {
      margin-left: 10px;
    } 
  </style>
  <div class="buttons">
    <a class="muted-link" href="https://github.com/TrusteeML/trustee" aria-label="GitHub"> 
        <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
            <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
        </svg>
        <span class="label">Github Repo</span>
    </a>
    <a class="muted-link" href="https://github.com/TrusteeML/emperor" aria-label="GitHub"> 
        <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
            <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
        </svg>
        <span class="label">Use Cases</span>
    </a>
    <a class="muted-link" href="https://github.com/TrusteeML/emperor/raw/main/docs/tech-report.pdf"aria-label="GitHub">
      <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
        <path d="M854.6 288.7c6 6 9.4 14.1 9.4 22.6V928c0 17.7-14.3 32-32 32H192c-17.7 0-32-14.3-32-32V96c0-17.7 14.3-32 32-32h424.7c8.5 0 16.7 3.4 22.7 9.4l215.2 215.3zM790.2 326L602 137.8V326h188.2zM633.22 637.26c-15.18-.5-31.32.67-49.65 2.96-24.3-14.99-40.66-35.58-52.28-65.83l1.07-4.38 1.24-5.18c4.3-18.13 6.61-31.36 7.3-44.7.52-10.07-.04-19.36-1.83-27.97-3.3-18.59-16.45-29.46-33.02-30.13-15.45-.63-29.65 8-33.28 21.37-5.91 21.62-2.45 50.07 10.08 98.59-15.96 38.05-37.05 82.66-51.2 107.54-18.89 9.74-33.6 18.6-45.96 28.42-16.3 12.97-26.48 26.3-29.28 40.3-1.36 6.49.69 14.97 5.36 21.92 5.3 7.88 13.28 13 22.85 13.74 24.15 1.87 53.83-23.03 86.6-79.26 3.29-1.1 6.77-2.26 11.02-3.7l11.9-4.02c7.53-2.54 12.99-4.36 18.39-6.11 23.4-7.62 41.1-12.43 57.2-15.17 27.98 14.98 60.32 24.8 82.1 24.8 17.98 0 30.13-9.32 34.52-23.99 3.85-12.88.8-27.82-7.48-36.08-8.56-8.41-24.3-12.43-45.65-13.12zM385.23 765.68v-.36l.13-.34a54.86 54.86 0 0 1 5.6-10.76c4.28-6.58 10.17-13.5 17.47-20.87 3.92-3.95 8-7.8 12.79-12.12 1.07-.96 7.91-7.05 9.19-8.25l11.17-10.4-8.12 12.93c-12.32 19.64-23.46 33.78-33 43-3.51 3.4-6.6 5.9-9.1 7.51a16.43 16.43 0 0 1-2.61 1.42c-.41.17-.77.27-1.13.3a2.2 2.2 0 0 1-1.12-.15 2.07 2.07 0 0 1-1.27-1.91zM511.17 547.4l-2.26 4-1.4-4.38c-3.1-9.83-5.38-24.64-6.01-38-.72-15.2.49-24.32 5.29-24.32 6.74 0 9.83 10.8 10.07 27.05.22 14.28-2.03 29.14-5.7 35.65zm-5.81 58.46l1.53-4.05 2.09 3.8c11.69 21.24 26.86 38.96 43.54 51.31l3.6 2.66-4.39.9c-16.33 3.38-31.54 8.46-52.34 16.85 2.17-.88-21.62 8.86-27.64 11.17l-5.25 2.01 2.8-4.88c12.35-21.5 23.76-47.32 36.05-79.77zm157.62 76.26c-7.86 3.1-24.78.33-54.57-12.39l-7.56-3.22 8.2-.6c23.3-1.73 39.8-.45 49.42 3.07 4.1 1.5 6.83 3.39 8.04 5.55a4.64 4.64 0 0 1-1.36 6.31 6.7 6.7 0 0 1-2.17 1.28z"></path>
      </svg>
      <span class="label">Tech Report</span>
    </a>
  </div>

Overview
-------------

Trustee is a framework to extract decision tree explanation from black-box ML models.

.. figure:: _static/flowchart.png
  :align: center
  :alt: Trustee Flowchart
  
  Standard AI/ML development pipeline extended by Trustee.
  

Getting Started
---------------
This section contains basic information and instructions to get started with Trustee.

Python Version
***************

Trustee supports Python >=3.7.

Install Trustee
***************

Use the following command to install Trustee:

.. code-block:: sh

    $ pip install trustee


Sample Code
*******************

.. code:: python
  
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
  dt, pruned_dt, agreement, reward = trustee.explain()
  dt_y_pred = dt.predict(X_test)

  print("Model explanation global fidelity report:")
  print(classification_report(y_pred, dt_y_pred))
  print("Model explanation score report:")
  print(classification_report(y_test, dt_y_pred))


Other Use Cases
*******************
For other examples and use cases of how Trustee can used to scrutinize ML models, listed in the table below, please check our `Use Cases repository <https://github.com/TrusteeML/emperor>`_.

.. table::
    :class: align-left
    
    ===================== ===========================================================================================================================================================
    Use Case              Description
    ===================== ===========================================================================================================================================================
    `heartbleed_case/`    Trustee application to a Random Forest Classifier for an Intrustion Detection System, trained with CIC-IDS-2017 dataset pre-computed features.
    `kitsune_case/`       Trustee application to Kitsune  model for anomaly detection in network traffic, trained with features extracted from Kitsune's Mirai attack trace.
    `iot_case/`           Trustee application to Random Forest Classifier to distguish IoT devices, trained with features extracted from the pcaps from the UNSW IoT Dataset.
    `moon_star_case/`     Trustee application to Neural Network Moon and Stars Shortcut learning toy example.
    `nprint_ids_case/`    Trustee application to the nPrintML AutoGluon Tabular Predictor for an Intrustion Detection System, also trained using pcaps from the CIC-IDS-2017 dataset.
    `nprint_os_case/`     Trustee application to the nPrintML AutoGluon Tabular Predictor for OS Fingerprinting, also trained using with pcaps from the CIC-IDS-2017 dataset.
    `pensieve_case/`      Trustee application to the Pensieve RL model for adaptive bit-rate prediction, and comparison to related work Metis.
    `vpn_case/`           Trustee application the 1D-CNN trained to detect VPN traffic trained with the ISCX VPN-nonVPN dataset.
    ===================== ===========================================================================================================================================================

Supported AI/ML Libraries
*************************

.. table::
    :class: align-left

    ==============  ===================
    Library         Supported  
    ==============  ===================
    `scikit-learn`  |:white_check_mark:|
    `Keras`         |:white_check_mark:|
    `Tensorflow`    |:white_check_mark:|
    `PyTorch`       |:white_check_mark:|
    `AutoGluon`     |:white_check_mark:|
    ==============  ===================

API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
  :maxdepth: 2

  api
  auto_examples/index


Citing Us
---------

.. code::

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

