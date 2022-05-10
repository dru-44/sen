import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Sentinel-imgpackage",
    version="0.0.1",
    author="Dru44",
    author_email="dhruthik28@gmail.com",
    license='MIT',
    description="Sentinel satellite image module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dru-44/sen",
    project_urls={
        "Bug Tracker": "https://github.com/dru-44/sen/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        'matplotlib',
        'numpy',
        'rasterio',
        'plotly',
        'earthpy',
        'tensorboard',
        'rich',
        'lightgbm',
        'seaborn',
    ],
    python_requires=">=3.6",
)
