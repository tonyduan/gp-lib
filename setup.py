import setuptools


setuptools.setup(
    name="gp-lib",
    version="0.0.2",
    author="Tony Duan",
    author_email="tonyduan@cs.stanford.edu",
    description="Lightweight Python library for GP regression.",
    long_description="Please see Github for full description.",
    long_description_content_type="text/markdown",
    url="https://github.com/tonyduan/gp-lib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
