# SynTwins

Implementation of synthetic accessible analog design with SynTwins developed by MICC group at SNU (contact: yousung@gmail.com).

## Contents

- [Developer](#developer)
- [OS Requirements](#os-requirements)
- [Python Dependencies](#python-dependencies)
- [Installation Guide](#installation-guide)
- [Reproduce the results](#reproduce-the-results)
- [Citation](#citation)

## Developer
Shuan Chen (shuan75@snu.ac.kr)<br>

## OS Requirements
This repository has been tested on both **Linux** and **Windows** operating systems.

## Python Dependencies
* Python (version >= 3.6) 
* Numpy (version >= 1.16.4) 
* RDKit (version >= 2019)

## Installation Guide
Create a virtual environment to run the code of SynTwins.<br>
This process usually takes few munites to complete.<br>
```
git clone https://github.com/snu-micc/SynTwins.git
cd SynTwins
conda create -c conda-forge -n rdenv python=3.6 -y
conda activate rdenv
```

## Reproduce the results
### Data preperation
The necessary files-including reaction templates, retro-reaction templates, and builing blocks- are provided in the `data` directory. You can replace the building blocks and reaction templates following the same format.
No futher preprocessing or training is needed to implement SynTwins.

### Synthetically accessible analog design
Try to make synthetically accessible analog with SynTwins following the examples in `Demo.ipynb`!
Test files used in this paper are all available at the `data/test` directory.

### Molecule Optimization
Run `run_optimization.py` to reproduce the molecule optimization results.
This experiment was largely reproduced using the script from the [mol-opt repsitory](https://github.com/wenhao-gao/mol_opt).
Note that the python package `tdc` needed to be installed to run this experiment. You can install it by
```
pip install PyTDC
```

## Citation
under review
