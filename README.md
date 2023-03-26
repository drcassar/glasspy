# Welcome to GlassPy
[![DOI](https://zenodo.org/badge/197668520.svg)](https://zenodo.org/badge/latestdoi/197668520)

GlassPy is a Python module for scientists working with glass materials.

![Screenshot](docs/source/logo/logo_text_small.png)

## What is it?
GlassPy's current focus is to provide an easy way to load SciGlass data and use GlassNet and ViscNet, two deep learning predictive models of glass and glass-forming liquid properties.

## How to install
The source code is available on GitHub at https://github.com/drcassar/glasspy.

Binary installers for the latest released version are available from the [Python Package Index] (https://pypi.org/project/glasspy/). To install GlassPy with pip run

```sh
pip install glasspy
```

Optionally, you can install GlassPy with additional features. Currently, the additional features are random forest models to improve GlassNet's prediction. Note that this adds about 1GB to the installation. To install GlassPy with all the extra stuff, run

```sh
pip install "glasspy[extra]"
```

Note that you need to have PyTorch and Lightning installed if you want to use the predictive models that come with GlassPy. These modules are not installed by default. Please read the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) and the [Lightning Installation Guide](https://lightning.ai/docs/pytorch/stable/#install-lightning) for more information.

## Development
GlassPy is under development. API changes are not only likely, but expected as development continues.

## How to cite
Daniel R. Cassar. drcassar/glasspy: GlassPy. Zenodo. https://doi.org/10.5281/zenodo.3930350

## GlassPy license
[GPL](https://github.com/drcassar/glasspy/blob/master/LICENSE)

GlassPy, Python module for scientists working with glass materials. Copyright (C) 2019-2023 Daniel Roberto Cassar

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
