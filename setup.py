from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='BambooTools',
    version='0.1.0',
    author='Konstantinos Maravegias',
    author_email='kwstas.maras@gmail.com',
    packages=find_packages(),
    url='http://pypi.python.org/pypi/BambooTools/',
    license='MIT',
    description='Useful Bamboo stuff.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "pandas >= 1.5.3"
    ],
)