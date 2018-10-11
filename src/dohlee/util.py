import os

def strip_ext(f):
    return os.path.splitext(os.path.basename(f))[0]
