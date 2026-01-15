
testmace:

environment.intent.yml: TESTMACE, the base environment i created with:

conda create --name testmace python=3.10.19 pip=25.3 -y 
conda install -y -c pytorch -c nvidia pytorch=2.5.1 pytorch-cuda=12.1
conda install NumPy SciPy 
pip install ase
conda install ipykernel
pip install mace-torch


# versions as read: 
NumPy version: 2.2.6
SciPy version: 1.15.3
ASE version: 3.27.0
mace-torch version: 0.3.14
mattersim version: 1.2.0





