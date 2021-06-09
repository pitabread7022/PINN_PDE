#!/bin/sh
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
pip install virtualenv
virtualenv -p /usr/bin/python3.6 PINN_env
source PINN_env/bin/activate
pip install numpy
pip install scipy
pip install matplotlib==2.2.3
pip install tensorflow==1.15.0
apt-get install python3-tk
pip install pyDOE
pip install latex