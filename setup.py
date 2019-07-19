from distutils.core import setup

setup(
    name='glasspy',
    version='0.1dev',
    author='Daniel R. Cassar',
    author_email='daniel.r.cassar@gmail.com',
    packages=['glasspy',],
    license='GPL',
    description='Python tools for glass science',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'scipy',],
)
