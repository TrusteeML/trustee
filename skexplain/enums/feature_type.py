""" Feature Type Enum """
from enum import Enum


class FeatureType(Enum):
    """ Types of features in training/testing datasets """
    CATEGORICAL = 0
    NUMERICAL = 1
    IDENTIFIER = 2
