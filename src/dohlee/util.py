import os
import collections

def strip_ext(f):
    return os.path.splitext(os.path.basename(f))[0]

def most_common_value(counter):
    return counter.most_common()[0][0]
