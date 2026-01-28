from setuptools import setup, find_packages

setup(
    name="circledetect",
    version="0.1.0",
    description="Python pipeline to track circular particles and transform images",
    author="fcichos",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    python_requires=">=3.7",
)
