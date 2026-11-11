#!/usr/bin/env bash
set -e
mkdir -p /kaggle/temp/mm
cd /kaggle/temp
curl -sS -L -o mm.tar.bz2 https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.10.0-0.tar.bz2
tar -xjf mm.tar.bz2 -C mm
export MAMBA_ROOT_PREFIX=/kaggle/temp/mroot
./mm/bin/micromamba create -y -q -n old -c conda-forge python=3.10 "pykep=2.6" pygmo numpy matplotlib
echo "ENV READY"