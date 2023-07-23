import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read().replace(
        "![Screenshot](docs/source/logo/logo_text_small.png)\n\n", ""
    )

setuptools.setup(
    name="glasspy",
    version="0.4.2",
    author="Daniel Roberto Cassar",
    author_email="daniel.r.cassar@gmail.com",
    description="Python module for scientists working with glass materials",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drcassar/glasspy",
    packages=setuptools.find_packages(),
    install_requires=[
        "pandas>=1.3",
        "lmfit>=1.0.0",
        "chemparse>=0.1.0",
        "scikit-learn==1.2.0",
        "compress_pickle>=2.1.0",
        "torch",
        "lightning>=2.0.0",
    ],
    extras_require={
        "extra": ["glasspy_extra"],
    },
    keywords="glass, non-crystalline materials",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 "
            "or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
    ],
    license="GPL",
    python_requires=">=3.9",
    include_package_data=True,
)
