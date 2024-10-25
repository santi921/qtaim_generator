# GBW analysis

## Overview / Installation
There are a few dependencies, namely: 
- json
- pathlib
- ORCA (specifically orca_2mkl)
- Multiwfn

Simply install this package by cloning the repo and running: 
```
pip install -e .
```

## Usage
There is one function that performs all the gbw analysis we need and saves results to jsons: 
<code> qtaim_gen.source.core.omol.gbw_analysis </code> 

In short, this function will create a file for converting gbws to wfn files, create jobs for multiwfn, runs those jobs, and then process the raw txts into jsons for easy processing. If we are worried about storage, there is an optional flag to delete wfn files after they are used. This function requires 3 arguments: 
    -folder: target directory with either .gbw or .wfn files and .inp
    -multiwfn_cmd: command for multiwfn.
    -orca_2mkl_cmd: command for orca_2mkl for conversion.

NOTA BENE:  orca_2mkl is sensitive to the version of orca your gbw is from, this means that there needs to be some logic to pass the correct orca_2mkl version for the current directory

## Environment Settings

Prior to running just set the following two variables: 
- <code> ulimit -s unlimited </code>
- <code> export OMP_STACKSIZE=64000000 </code>


