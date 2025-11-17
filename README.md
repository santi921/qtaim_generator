# Generator

<img src="https://github.com/santi921/qtaim_generator/blob/main/qtaim_gen/notebooks/TOC.png" width=50% height=50%>

A package to perform post-processing on molecules, reactions, and (soon) periodic systems. It's a wrapper around Multiwfn that handles high-throughput workflows and can compute a rich set of descriptors including QTAIM, partial charges, several bonding schemes, etc. I am currently overhauling the package away from being QTAIM-first and instead integrating many descriptors in tandem. I will be sticking with the JSON file format for QTAIM and the "full" feature implementations so running calculations will remain the same, gathering and parsing utilities for ML, however, will change. Here's a quick status about what is available for QTAIM/full at the moment:

### QTAIM
- [x] Reaction-level feature generation and processing
- [x] Molecule-level feature generation and processing
- [x] Post-processing to LMDBs in DGL for ML


### Full
- [x] Molecule-level feature generation and processing
- [ ] Reaction-level feature generation and processing
- [x] High-throughput runners using python MP
- [x] High-throughput runners using parsl 
- [ ] Post-processing to LMDBs for ML (ongoing)
- [ ] Quacc integration (ongoing)


Simply install this package by cloning the repo and running: 
```
pip install -e .
```

# Overview - QTAIM Usage

We can use QTAIM to define bonds in a system as well as define a rich set of descriptors for machine learning. With a few scripts you can get to generating QTAIM-informatics for analysis and machine learning tasks. Currently, this package supports BondNet(<a href="https://github.com/santi921/bondnet">BonDNet</a>) (for reaction-property predicton) and <a href="https://github.com/santi921/qtaim_embed">QTAIM-Embed</a> and ChemProp <a href="[https://github.com/santi921/qtaim_embed](https://github.com/chemprop/chemprop)">QTAIM-Embed</a> for molecular machine learning tasks. Note that the Chemprop implementation currently only supports atom-level QTAIM descriptors. 

## Overview
To get started you will need to decide a few things: 
1) DFT Software: We currently have input file writers for Orca though creating custom writers for other software should be easy to integrate. For ORCA, we add a few options such a relativistic corrections and atom-specific basis sets. See the <a href="[https://github.com/santi921/qtaim_generator/blob/main/qtaim_gen/source/scripts/options_qm.json]">example JSON</a>  for more options 
2) QTAIM software: The implementation with Critic2 works but is relatively experimental and we suggest you use Multiwfn as it yields a richer set of QTAIM features. 
3) Level of theory: QTAIM is pretty resistant to low levels of theory. Take care, however, when your dataset contains metals (especially heavy metals where this assertion is  less tested). 


## Usage
Three scripts will be needed to generate QTAIM features readily formatted for your dataset. These scripts generate job files, run jobs, and parse outputs to a single json, respectively. For the following we will assume you have a properly formatted json/pickle/bson and will return to this later. 

1) <code> create_files.py </code> - generates input files for DFT and QTAIM jobs and has severate arguments:
    - <code> -reaction </code> : specifies whether the dataframe 
    - <code> -parser </code> Multiwfn or Critic2
    - <code> -file </code> specifies the dataset file
    - <code> -root </code> specifies where to write job files
    - <code> -options_qm_file </code> options for your electronic structure job
    - <code> --molden_sub </code> whether to use <code> orca_2mkl </code> to convert the a gbw to a .molden.input file prior to Multiwfn. Use this if you intend on using ECPs.
2) <code> run.py </code> - runs DFT and QTAIM jobs in selected folder
    - <code> -redo_qtaim </code> - whether to clear QTAIM results file and redo 
    - <code> -just_dft </code> - whether to scriptly run DFT jobs
    - <code> --reactions </code> : specifies whether the root folder contains reaction or molecule jobs
    - <code> -dir_active </code> - root folder of QTAIM/DFT jobs
    - <code> -orca_path </code> - path to ORCA executable
    - <code> -num_threads </code> - number of threads for DFT jobs
    - <code> -folders_to_crawl </code> - how many folders to check for complete jobs
3) <code> parse_data.py </code> takes DFT/QTAIM output files and merges QTAIM data into a the original data structure:
    - <code> --root </code> root folder of QTAIM/DFT jobs
    - <code> --file_in </code> - input dataframe used to construct QTAIM/DFT jobs
    - <code> --impute </code> - whether or not to fill in missing values with mean values from computed statistics
    - <code> --file_out </code> - where to write to
    - <code> --reaction </code> - where your data is a reaction dataset
    - <code> --update_bonds_w_qtaim </code> -whether to overwrite existing bond definitions
    - <code> -define_bonds </code> - method ("distances" or "qtaim") of determining bonds
   
## Extra Scripts
1) <code> parse_stop.py </code> computes and prints statistics of QTAIM values in selected folder
2) <code> check_res_rxn_json.py </code> checks the number of complete jobs for reaction QTAIM run
3) <code> check_res_wfn.py </code> checks the number of complete jobs for molecular QTAIM run
4) <code> folder_xyz_molecules_to_pkl.py </code> converts a folder of xyz files into a single dataset for subsequent QTAIM generation.

## Data Structure
Jsons, pkls, and bson can all be parsed. 



# Overview - Full Usage
# TODO


 ## Citation 
 If you use this package please cite the following, thanks!
 
 @Article{D4DD00057A,
author ="Vargas, Santiago and Gee, Winston and Alexandrova, Anastassia",
title  ="High-throughput quantum theory of atoms in molecules (QTAIM) for geometric deep learning of molecular and reaction 
properties",
journal  ="Digital Discovery",
year  ="2024",
volume  ="3",
issue  ="5",
pages  ="987-998",
publisher  ="RSC",
doi  ="10.1039/D4DD00057A",
url  ="http://dx.doi.org/10.1039/D4DD00057A"
}



## install 
