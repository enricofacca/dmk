========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/dmk/badge/?style=flat
    :target: https://dmk.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/enricofacca/dmk/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/enricofacca/dmk/actions

.. |requires| image:: https://requires.io/github/enricofacca/dmk/requirements.svg?branch=main
    :alt: Requirements Status
    :target: https://requires.io/github/enricofacca/dmk/requirements/?branch=main

.. |codecov| image:: https://codecov.io/gh/enricofacca/dmk/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/enricofacca/dmk

.. |version| image:: https://img.shields.io/pypi/v/dmk.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/dmk

.. |wheel| image:: https://img.shields.io/pypi/wheel/dmk.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/dmk

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/dmk.svg
    :alt: Supported versions
    :target: https://pypi.org/project/dmk

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/dmk.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/dmk

.. |commits-since| image:: https://img.shields.io/github/commits-since/enricofacca/dmk/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/enricofacca/dmk/compare/v0.0.0...main



.. end-badges

Dynamical MoKantorovich solver

* Free software: MIT license

Installation
============

::

    pip install dmk

You can also install the in-development version with::

    pip install https://github.com/enricofacca/dmk/archive/main.zip


Documentation
=============


https://dmk.readthedocs.io/


Development
===========

To run all the tests run::

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
