from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

project_urls = {
    "Documentation": "https://github.com/KwstasMCPU/BambooTools",
    "Changelog": "https://github.com/KwstasMCPU/BambooTools/releases",
    "Source": "https://github.com/KwstasMCPU/BambooTools",
}

setup(
    name="BambooTools",
    version="0.4.0",
    author="Konstantinos Maravegias",
    author_email="kwstas.maras@gmail.com",
    packages=find_packages(),
    url="https://github.com/KwstasMCPU/BambooTools",
    license="MIT",
    description="Pandas extension to enchance your data analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["pandas>=1.5.3"],
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "twine",
            "seaborn",
            "build",
            "pre-commit==3.5.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
    keywords=[
        "BambooTools",
        "pandas",
        "pandas extensions",
        "data analysis",
        "data science",
        "analytics",
    ],
    project_urls=project_urls,
    python_requires=">=3.8",
)
