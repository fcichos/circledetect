"""Setup script for CircleDetection package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / 'README.md'
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')
else:
    long_description = "Microparticle tracking for dark field microscopy"

setup(
    name='circledetection',
    version='1.0.0',
    author='CircleDetection Team',
    description='Microparticle tracking for dark field microscopy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/CircleDetection',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'opencv-python>=4.5.0',
        'numpy>=1.20.0',
        'matplotlib>=3.5.0',
        'scipy>=1.7.0',
        'click>=8.0.0',
    ],
    entry_points={
        'console_scripts': [
            'circledetect=circledetection.cli:cli',
        ],
    },
    include_package_data=True,
)
