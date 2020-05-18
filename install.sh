#!/bin/bash

echo " ***** INSTALLING REQUIRED PACKAGES...*****"
pip install -r requirements.txt

echo " ***** INSTALLING Autoencoders PACKAGE...*****"
python setup.py install

