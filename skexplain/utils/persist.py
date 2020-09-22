import os
import traceback

from joblib import dump, load


def load_model(path):
    """ Loads model weights, if they exist """
    if path and os.path.isfile(path):
        try:
            model = load(path)
            return model
        except:
            traceback.print_exc()
            return None
    return None


def save_model(model, path):
    """ Saves model weights """
    return dump(model, path)
