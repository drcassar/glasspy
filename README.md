# Welcome to GlassPy
GlassPy is a Python module for scientists working with glass materials.

![Screenshot](doc/logo/logo_text_small.png)

## What is it?
The aim is to provide classes and functions written in Python for the materials scientists working with glass. The hope is that with an open and collaborative project, we can build a reliable toolset to support faster and reproducible research on glass materials.

## How to install
The source code is hosted on GitHub at: https://github.com/drcassar/glasspy.

Binary installers for the latest released version are available at the [Python package index](https://pypi.org/project/glasspy/). To install GlassPy run

```sh
pip install glasspy
```

## Development
GlassPy was born as a personal tool back in 2013 when I started coding with Python. It is based on a colection of MATLAB code that I wrote for the Glass State graduate course of 2010 and for the numerical analysis for my PhD.

Right now, I'm sorting all my code and adequately documenting it to build this Python module. My personal objective is to increase the reproducibility of my research and hopefully be useful for researchers working with glass science.

## Roadmap
This repository is in its infancy. The current version is 0.1.dev1, which means that it is not intended for public use right now.

My objective for version 0.1 is to have working classes for the regression of nucleation density data.

## Documentation
There is no documentation right now, but all the functions have a detailed docstring.
Examples in the form of Jupyter Notebooks can be found [here](doc/examples/).

## Dependencies
- [Python 3.6+](https://www.python.org/)
- [NumPy](https://www.numpy.org)
- [SciPy](https://www.scipy.org/)
- [Pandas](https://pandas.pydata.org/)
- [lmfit](https://lmfit.github.io/lmfit-py/)

## Other python repositories for glass science
- [RelaxPy](https://github.com/Mauro-Glass-Group/RelaxPy) - Module to compute glass relaxation kinetics.
- [PyGlass](https://github.com/jrafolsr/PyGlass) - Module to simulate the specific heat signature of glasses with a specified thermal treatment following the Tool-Narayanaswamy-Moynihan model.

## License
[GPL](LICENSE)

GlassPy, Python module for scientists working with glass materials. Copyright (C) 2019 Daniel Roberto Cassar

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
