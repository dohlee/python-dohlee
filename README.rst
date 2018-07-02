========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |coveralls| |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|

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

My personal python utility library. Currently on version v0.1.5.

* Free software: MIT license

Installation
============

::

    pip install dohlee

Documentation
=============

https://python-dohlee.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
