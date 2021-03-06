.. start-badges

|docs| |version| |travis| |coveralls|

.. |docs| image:: https://readthedocs.org/projects/python-dohlee/badge/?style=flat
    :target: https://readthedocs.org/projects/python-dohlee
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/dohlee/python-dohlee.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/dohlee/python-dohlee

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/dohlee/python-dohlee?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/dohlee/python-dohlee

.. |requires| image:: https://requires.io/github/dohlee/python-dohlee/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/dohlee/python-dohlee/requirements/?branch=master

.. |coveralls| image:: https://coveralls.io/repos/dohlee/python-dohlee/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/dohlee/python-dohlee

.. |codecov| image:: https://codecov.io/github/dohlee/python-dohlee/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/dohlee/python-dohlee

.. |version| image:: https://img.shields.io/pypi/v/dohlee.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/dohlee

.. |wheel| image:: https://img.shields.io/pypi/wheel/dohlee.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/dohlee

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/dohlee.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/dohlee

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/dohlee.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/dohlee


.. end-badges
======
dohlee
======

My personal python utility library. Currently on version v0.1.15.

Installation
============

::

    pip install dohlee

Documentation
=============

https://python-dohlee.readthedocs.io/

Examples
========

::

    import itertools
    import numpy as np
    import dohlee.plot as plot; plot.set_style()
    from collections import Counter

    data = Counter()
    c_contexts = [p + 'C' + n for (p, n) in itertools.product('ACGT', 'ACGT')]
    t_contexts = [p + 'T' + n for (p, n) in itertools.product('ACGT', 'ACGT')]
    c_alts, t_alts = 'AGT', 'ACG'

    for context, alt in itertools.product(c_contexts, c_alts):
        data[(context, alt)] = np.random.randint(1, 30)
    for context, alt in itertools.product(t_contexts, t_alts):
        data[(context, alt)] = np.random.randint(1, 30)

    ax = plot.get_axis(figsize=(20.4, 3.4))
    plot.mutation_signature(data, ax=ax)
    plot.set_suptitle('Mutational signatures.')

.. image:: img/mutation_signature.png
    :width: 800
