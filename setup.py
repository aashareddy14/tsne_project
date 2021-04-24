import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TSNE-reddy-beckner",
    version="0.0.2",
    author="Aasha Reddy, Madeleine Beckner",
    author_email="aashareddy14@gmail.com",
    description="Implementation of t-Distributed Stochastic Neighbor Embedding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aashareddy14/tsne_project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)