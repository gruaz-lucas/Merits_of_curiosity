from setuptools import setup, find_packages

setup(
    name='Merits_of_curiosity',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.1.3',
        'networkx>=2.0',
        'matplotlib>=3.9',
        'gymnasium>=1.0.0',
        'scipy>=1.14.1',
    ]
)