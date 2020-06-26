from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ilaff",
    version="0.1.0",
    description="Prototype of library for statistical analysis and fitting of lattice data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ILAFF/ILAFF",
    author="ILAFF",
    author_email="someone@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        # "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="lattice analysis qcd",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6, <4",
    install_requires=[
        "dataclasses;python_version<'3.7'",
        "numpy>=1.13",
    ],
    extras_require={"dev": ["sphinx", "sphinx_rtd_theme"],},
    project_urls={
        "Bug Reports": "https://github.com/ILAFF/ILAFF/issues",
        "Source": "https://github.com/ILAFF/ILAFF/",
    },
)
