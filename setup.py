from setuptools import find_packages, setup
from autoencoders import __version__

VERSION = __version__ 

setup(
    name='autoencoders',
    packages=find_packages(),
    version=VERSION,
    description='autoencoders',
    author='andrea.karlova@gmail.com',
    license='',
)
