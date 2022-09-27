## -- external modules may need to add for functionality --
cd /
git clone https://github.com/celiibrendan/python_tools  
git clone https://github.com/celiibrendan/machine_learning_tools.git  
git clone https://github.com/celiibrendan/neuron_morphology_tools.git  
git clone https://github.com/celiibrendan/pytorch_tools.git  

"""
Would then add to your path as follows: 
from os import sys  
sys.path.append("/python_tools/python_tools")  
sys.path.append("/machine_learning_tools/machine_learning_tools/")  
sys.path.append("/pytorch_tools/pytorch_tools/")  
sys.path.append("/neuron_morphology_tools/neuron_morphology_tools/")
"""