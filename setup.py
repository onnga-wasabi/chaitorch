from setuptools import setup, find_packages

setup(
    name="chaitorch",
    version="0.1",
    description="Pytroch wrapper for those who want to coding w/o 'global'",
    url="https://github.com/onnga-wasabi/pytorch-tutorial.git",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
    ]
)
