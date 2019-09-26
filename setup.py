import setuptools

with open("README_PyPI.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='glasspy',
    version='0.1.dev1',
    author='Daniel Roberto Cassar',
    author_email='daniel.r.cassar@gmail.com',
    description='Python module for scientists working with glass materials',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drcassar/glasspy",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.1', 'scipy>=0.19', 'pandas>=0.24.0', 'lmfit>=0.9.13'
    ],
    keywords='glass, non-crystalline materials',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
    ],
    license='GPL',
    python_requires='>=3.6',
)
