import os
import sys

MB_ROOT = os.path.abspath(os.path.dirname(__file__))

MB_LIB = os.path.join(MB_ROOT, "lib")
MB_NOTEBOOKS = os.path.join(MB_ROOT, "notebooks")
MB_SRC = os.path.join(MB_ROOT, "src")

MB_IMG = os.path.join(MB_ROOT, "img")

MB_DATA = os.path.join(MB_ROOT, "data")
MB_RAW = os.path.join(MB_DATA, "raw")
MB_TMP = os.path.join(MB_DATA, "tmp")
MB_PROCESSED = os.path.join(MB_DATA, "processed")

sys.path.extend([
    MB_LIB, 
    MB_NOTEBOOKS,
    MB_SRC
])