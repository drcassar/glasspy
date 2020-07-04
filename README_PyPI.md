# Welcome to GlassPy
GlassPy is a Python module for scientists working with glass materials.

## What is it?
The aim is to provide classes and functions written in Python for materials scientists working with glass and non-crystalline materials. The hope is that with an open and collaborative project, we can build a reliable toolset to support faster and reproducible research on this topic.

## How to install
The source code is hosted on GitHub at https://github.com/drcassar/glasspy.

Binary installers for the latest released version are available at the [Python Package Index](https://pypi.org/project/glasspy/). To install GlassPy using pip run

```sh
pip install glasspy
```

To install the latest development version of GlassPy run

```sh
pip install --upgrade git+git://github.com/drcassar/glasspy
```
## Development
GlassPy was born as a personal tool back in 2013 when I started coding with Python. It is based on a collection of MATLAB code that I wrote for the Glass State graduate course of 2010 and the numerical analysis during my PhD.

Right now, I'm sorting all my code and adequately documenting it to build this Python module. My personal objective is to increase my research's reproducibility and hopefully be useful for researchers working in the field of glass science.

## Documentation
There is no documentation right now, but all the functions have detailed docstring.

Some examples are provided as notebooks in Google Colab (they run in the cloud, so you can test GlassPy without the need of installing anything on your machine):

- [Finding the Kauzmann Temperature](https://colab.research.google.com/drive/1WAVddrOSDG7_pydV9g-WBfBisKQrMVeF)
- [Regression of nucleation density data](https://colab.research.google.com/drive/139ke_S3eNq6MvJ3A_GnTWtlI8rn2QwKf)
- [Regression of viscosity data](https://colab.research.google.com/drive/1U_xSg4H-de9QDlXChCN9Tsda8VlBlbAI)
- [Nucleation kinetic model](https://colab.research.google.com/drive/1mRAr0HNXcO5YaAFktir-Kb9K4ZUdiXk_)

## Dependencies
- [Python 3.6+](https://www.python.org/)
- [NumPy](https://www.numpy.org)
- [SciPy](https://www.scipy.org/)
- [Pandas](https://pandas.pydata.org/)
- [lmfit](https://lmfit.github.io/lmfit-py/)
- [chemparse](https://pypi.org/project/chemparse/)

## Other python repositories for glass science
- [RelaxPy](https://github.com/Mauro-Glass-Group/RelaxPy) - Module to compute glass relaxation kinetics.
- [PyGlass](https://github.com/jrafolsr/PyGlass) - Module to simulate the specific heat signature of glasses with a specified thermal treatment following the Tool-Narayanaswamy-Moynihan model.

## SciGlass database licence
[ODbL](https://github.com/drcassar/glasspy/blob/master/glasspy/data/datafiles/LICENCE_sciglass)

ODC Open Database License (ODbL)

Copyright (c) 2019 EPAM Systems

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## GlassPy license
[GPL](https://github.com/drcassar/glasspy/blob/master/LICENSE)

GlassPy, Python module for scientists working with glass materials. Copyright (C) 2019-2020 Daniel Roberto Cassar

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
