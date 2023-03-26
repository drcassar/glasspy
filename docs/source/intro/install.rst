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

    pip install "glasspy[extra]"

To install the latest development version of GlassPy run::

    pip install --upgrade git+git://github.com/drcassar/glasspy@dev

Note that you will need to have PyTorch and Lightning installed if you want to use the predictive models that come with GlassPy. These modules are not installed by default. Please read the `PyTorch installation guide`_ and the `Lightning installation guide`_.

.. _PyTorch installation guide: https://pytorch.org/get-started/locally/
.. _Lightning installation guide: https://lightning.ai/docs/pytorch/stable/#install-lightning

GlassPy is a Python module for researchers working with `glass materials`_ and


.. _Python Package Index: https://pypi.org/project/glasspy/
