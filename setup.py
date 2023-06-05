def get_install_requires(filepath=None):
    if filepath is None:
        filepath = "./"
    """Returns requirements.txt parsed to a list"""
    fname = Path(filepath).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    return targets

def external_git():
    return [
        "git+https://github.com/bacelii/python_tools.git"
    ]
    
def install_require_git():
    return []

def get_links():
    return []

from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='graph_tools', # the name of the package, which can be different than the folder when using pip instal
    version='1.0.0',
    description='',
    author='Brendan Celii',
    author_email='brendanacelii',
    packages=find_packages(),  #teslls what packages to be included for the install
    install_requires=get_install_requires() + install_require_git(), #external packages as dependencies
    dependency_links = get_links(),
    # if wanted to install with the extra requirements use pip install -e ".[interactive]"
    extras_require={
        #'interactive': ['matplotlib>=2.2.0', 'jupyter'],
    },
    
    # if have a python script that wants to be run from the command line
    entry_points={
        #'console_scripts': ['pipeline_download=Applications.Eleox_Data_Fetch.Eleox_Data_Fetcher_vp1:main']
    },
    scripts=[], 
    
)

