#!/bin/bash

sbatch --mem=400m -c1 --time=12:0:0 -o "slurm_test$$.output" test.sh
