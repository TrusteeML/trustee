from copy import Error
import os
import zipfile
import traceback

from joblib import dump, load


def load_model(path):
    """Loads model weights, if they exist"""
    file_path = path
    if zipfile.is_zipfile(path):
        file_dir = os.path.dirname(path)
        file_path = path.replace(".zip", "")
        zip_ref = zipfile.ZipFile(path)  # create zipfile object
        zip_ref.extractall(file_dir)  # extract file to dir
        zip_ref.close()  # close file

    if file_path and os.path.isfile(file_path):
        try:
            model = load(file_path)
            if zipfile.is_zipfile(path):
                os.remove(file_path)  # delete unzipped file

            return model
        except Error:
            traceback.print_exc()
            return None
    return None


def save_model(model, path):
    """Saves model weights"""
    return dump(model, path)
