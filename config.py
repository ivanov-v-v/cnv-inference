import os.path

__all__ = ["ROOT_DIR", "LIB_DIR", "SRC_DIR", "DATA_DIR"]

MB_ROOT = os.path.abspath(os.path.dirname(__file__))

MB_LIB = os.path.join(MB_ROOT, "lib")
MB_NOTEBOOKS = os.path.join(MB_ROOT, "notebooks")
MB_SRC = os.path.join(MB_ROOT, "src")

MB_DATA = os.path.join(MB_ROOT, "data")
MB_RAW = os.path.join(MB_DATA, "raw")
MB_TMP = os.path.join(MB_DATA, "tmp")
MB_PROCESSED = os.path.join(MB_DATA, "processed")
