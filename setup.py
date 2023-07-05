from pathlib import Path
from setuptools import setup, find_packages
from typing import List


"""
Explanation of how to get the git dependencies: 

https://stackoverflow.com/questions/32688688/how-to-write-setup-py-to-include-a-git-repository-as-a-dependency/54701434#54701434

"""
def get_links():
    return [
        #"python_tools @ git+https://git@github.com/bacelii/python_tools.git"
    ]

def get_install_requires(filepath=None):
    if filepath is None:
        filepath = "./"
    """Returns requirements.txt parsed to a list"""
    fname = Path(filepath).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
            
    targets += get_links()
    return targets

def get_dependencies():
    return [
        #"https://github.com/bacelii/python_tools.git"
    ]

setup(
    name='graph_tools', # the name of the package, which can be different than the folder when using pip instal
    version='1.0.0',
    description='',
    author='Brendan Celii',
    author_email='brendanacelii',
    packages=find_packages(),  #teslls what packages to be included for the install
    install_requires=get_install_requires(), #external packages as dependencies
    # dependency_links = get_dependencies(),
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

