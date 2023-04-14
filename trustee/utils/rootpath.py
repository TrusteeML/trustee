"""
    This code was copied from the python-rootpath project linked below. 
    We copied this snippet of code with no intention of stealing their code, 
    but to fix an install issue due rootpath depending on codecov.

    To avoid further issues like this in the future, it seemed like a good idea 
    to just incorporate the specific function we needed from the project.

    * python-rootpath:
        - https://github.com/grimen/python-rootpath
    * linked issues:
        - https://github.com/TrusteeML/trustee/issues/2
        - https://community.codecov.com/t/codecov-yanked-from-pypi-all-versions/4259/11
"""
# =========================================
#       IMPORTS
# --------------------------------------

import sys
import os
import re
import six

from os import path, listdir


# =========================================
#       CONSTANTS
# --------------------------------------

DEFAULT_PATH = "."
DEFAULT_ROOT_FILENAME_MATCH_PATTERN = ".git|requirements.txt"


# =========================================
#       FUNCTIONS
# --------------------------------------


def detect(current_path=None, pattern=None):

    """
    Find project root path from specified file/directory path,
    based on common project root file pattern.

    Examples:

        import rootpath

        rootpath.detect()
        rootpath.detect(__file__)
        rootpath.detect('./src')

    """

    current_path = current_path or os.getcwd()
    current_path = path.abspath(path.normpath(path.expanduser(current_path)))
    pattern = pattern or DEFAULT_ROOT_FILENAME_MATCH_PATTERN

    if not path.isdir(current_path):
        current_path = path.dirname(current_path)

    def find_root_path(current_path, pattern=None):
        if isinstance(pattern, six.string_types):
            pattern = re.compile(pattern)

        detecting = True

        found_more_files = None
        found_root = None
        found_system_root = None

        file_names = None
        root_file_names = None

        while detecting:
            file_names = listdir(current_path)
            found_more_files = bool(len(file_names) > 0)

            if not found_more_files:
                detecting = False

                return None

            root_file_names = filter(pattern.match, file_names)
            root_file_names = list(root_file_names)

            found_root = bool(len(root_file_names) > 0)

            if found_root:
                detecting = False

                return current_path

            found_system_root = bool(current_path == path.sep)

            if found_system_root:
                return None

            system_root = sys.executable

            while os.path.split(system_root)[1]:
                system_root = os.path.split(system_root)[0]

            if current_path == system_root:
                return None

            current_path = path.abspath(path.join(current_path, ".."))

    return find_root_path(current_path, pattern)
