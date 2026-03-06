## Installation

### 1. Clone the repository

From a Linux terminal, download the repository with:

```
git clone https://github.com/MintaAriel/ASE-Evolutionary-Algorithm.git
cd ASE-Evolutionary-Algorithm
```

### 2. Activate your conda environment

```
conda activate ase_env
```

Your terminal should look similar to:

```
(ase_env) vito@fedora:~$
```

### 3. Install the package

Run the following command in the directory where this `README.md` file is located:

```
pip install -e .
```

Installing the package in **editable mode** allows you to modify the source code without reinstalling it each time. Any
changes made inside the `src/` directory will automatically be available when running the scripts.
