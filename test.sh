#!/bin/bash

#SBATCH --exclude=cb-[05-20],eye-[01-04],wadi-[01-05],gsm-[01-04],sm-[01-04],sm-[07-08],sm-[15-20]

onnx_versions=(1.10.2)
rt_versions=(1.10.0)

echo "pid = $$"

for t in ${onnx_versions[@]}; do
    for s in ${rt_versions[@]}; do
    python3 -m venv env$$
    source env$$/bin/activate
    echo "Environment $VIRTUAL_ENV activated"
    pip install --upgrade pip
    pip install onnx==$t
    pip install onnxruntime==$s
    pip freeze
    python test.py >test$$.output
    echo "exit code is $?"
    deactivate
    echo "Environment onnx$t-rt$s deactivated"
    rm -r env$$
    done
done