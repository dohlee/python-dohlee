import dohlee.util as util


def given_filename_with_single_dot_splitext_test():
    filename = '/tmp/a/b/c/d/e/filename.png'
    assert util.stripext(filename) == 'filename'


def given_filname_with_multiple_dots_splitext_test():
    filename = '/tmp/a/b/c/d/e/f.i.l.e.n.a.m.e.png'
    assert util.stripext(filename) == 'f.i.l.e.n.a.m.e'
