import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))

LIB = os.path.join(ROOT, "lib")
NOTEBOOKS = os.path.join(ROOT, "notebooks")
SCRIPTS = os.path.join(ROOT, "scripts")
SRC = os.path.join(ROOT, "src")
TESTS = os.path.join(ROOT, "tests")

IMG = os.path.join(ROOT, "img")

DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
TMP = os.path.join(DATA, "tmp")
PROCESSED = os.path.join(DATA, "processed")

sys.path.extend([
    LIB, 
    NOTEBOOKS,
    SRC,
    TESTS
])
