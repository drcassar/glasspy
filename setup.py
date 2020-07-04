import setuptools

with open("README_PyPI.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='glasspy',
    version='0.3',
    author='Daniel Roberto Cassar',
    author_email='contact@danielcassar.com.br',
    description='Python module for scientists working with glass materials',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drcassar/glasspy",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.16',
        'scipy>=1.2',
        'pandas>=1.0.0',
        'lmfit>=1.0.0',
        'chemparse>=0.1.0',
    ],
    keywords='glass, non-crystalline materials',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
    ],
    license='GPL',
    python_requires='>=3.6',
    include_package_data=True,
)
