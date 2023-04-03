.. _intro-install:

==================
Installation guide
==================

.. _faq-python-versions:

Supported Python versions
=========================

GlassPy requires Python 3.9+.


.. _faq-install:

Installing GlassPy
==================

Binary installers for the latest released version are available from the `Python
Package Index`_. To install GlassPy and all its necessary dependencies, use
pip run::

    pip install glasspy

Optionally, you can install GlassPy with additional features. Currently, the additional features are Random Forest models to complement GlassNet. But be careful! This adds about 1 Gb to the installation. To install GlassPy with all the extra stuff, run::

    pip install --upgrade git+https://github.com/drcassar/glasspy_extra

.. _Python Package Index: https://pypi.org/project/glasspy/
