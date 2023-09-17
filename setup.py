from setuptools import setup, find_packages

with open("README.txt", 'r') as f:
    long_description = f.read()

setup(
    name='BambooTools',
    version='0.1.0',
    author='Konstantinos Maravegias',
    author_email='kwstas.maras@gmail.com',
    packages=find_packages(),
    #scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    url='http://pypi.python.org/pypi/TowelStuff/',
    license='MIT',
    description='Useful towel-related stuff.',
    long_description=long_description,
    install_requires=[
        "pandas >= 1.5.3"
    ],
)