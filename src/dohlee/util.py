import os

def stripext(f):
    return os.path.splitext(os.path.basename(f))[0]
