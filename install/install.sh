conda create --name ecint python==3.8 -y
conda activate ecint
sudo apt install postgresql
sudo apt install rabbitmq-server
git clone https://github.com/chenggroup/aiida-cp2k.git
git clone https://github.com/chenggroup/aiida-deepmd.git
git clone https://github.com/chenggroup/aiida-lammps.git
git clone https://github.com/chenggroup/ecint.git
pip install aiida --ignore-installed PyYAML
cd aiida-cp2k
pip install -e . --no-deps
cd ..
cd aiida-deepmd
pip install -e . --no-deps
cd ..
cd aiida-lammps
pip install -e . --no-deps
cd ..

