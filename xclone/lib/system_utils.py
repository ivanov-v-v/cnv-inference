import inspect
import logging
import pickle

from typing import Iterable

import numpy as np
import pandas as pd

def log_assert(bool_, message="", logger=None, logger_name="", verbose=False):
    """Use this as a replacement for assert if you want the failing of the
    assert statement to be logged."""
    if logger is None:
        logger = logging.getLogger(logger_name)
    try:
        assert bool_, message
    except AssertionError:
        # construct an exception message from the code of the calling frame
        last_stackframe = inspect.stack()[-2]
        source_file, line_no, func = last_stackframe[1:4]
        source = "Traceback (most recent call last):\n" + \
            '  File "%s", line %s, in %s\n    ' % (source_file, line_no, func)
        if verbose:
            # include more lines than that where the statement was made
            source_code = open(source_file).readlines()
            source += "".join(source_code[line_no - 3:line_no + 1])
        else:
            source += last_stackframe[-2][0].strip()
        logger.debug("%s\n%s" % (message, source))
        raise AssertionError("%s\n%s" % (message, source))


def pickle_dump(obj, path):
    with open(path, "wb") as outfile:
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.SparseDataFrame):
            obj.reset_index(drop=True, inplace=True)
        pickle.dump(obj, outfile, protocol=4)

        
def pickle_load(path):
    with open(path, "rb") as infile:
        return pickle.load(infile)


def is_sorted(iterable: Iterable):
    return np.all(iterable == sorted(iterable))


def load_cookiecutter_info(project_config):
    return {key : project_config.__dict__[key] 
            for key in ["ROOT", 
                        "DATA", "RAW", "TMP", "PROCESSED", 
                        "IMG", 
                        "SRC", "SCRIPTS", "TESTS", "LIB", "NOTEBOOKS"]}