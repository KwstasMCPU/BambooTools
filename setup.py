from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='BambooTools',
    version='0.2.0',
    author='Konstantinos Maravegias',
    author_email='kwstas.maras@gmail.com',
    packages=find_packages(),
    url='https://github.com/KwstasMCPU/BambooTools',
    license='MIT',
    description='Useful Bamboo stuff.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "pandas >= 1.5.3"
        ],
    extras_require={
        "dev": ["pytest", "twine", "seaborn"]
        },
    classifiers=["Development Status :: 3 - Alpha",
                 "Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License"
                 ],
    python_requires='>=3.8'
)
