import setuptools


__version__ = "0.0.1"


setuptools.setup(
    name='errodactyl',
    version=__version__,
    description='Sparse error detection inference engine',
    author='Nicholas Turner',
    author_email='nturner@cs.princeton.edu',
    url='https://github.com/seung-lab/errodactyl',
    packages=setuptools.find_packages(),
)
