#!/bin/sh

folder=$(pwd)

mkdir -p ${folder}/plots
mkdir -p ${folder}/results

python run_simulation.py "$@"