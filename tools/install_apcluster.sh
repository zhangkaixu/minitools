#!/bin/sh
\rm apcluster -r -f
mkdir apcluster
cd apcluster
wget http://www.psi.toronto.edu/affinitypropagation/software/apcluster_linux64.zip
unzip apcluster_linux64.zip
chmod +x apcluster
cd ..

