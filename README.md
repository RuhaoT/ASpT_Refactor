# ASpT_Refactor
## Introduction
This is a repository for the refactor of ASpT.
You can found the original ASpT repository [here](http://gitlab.hpcrl.cse.ohio-state.edu/chong/ppopp19_ae)
This repository is still under construction.

## Installation and Usage
### Install ASpT_Refactor
```shell
git clone #TODO
```

### Test Job: Verify web-Google and web-Stanford matrix
This test job is an functional verification of ASpT SpMM algorithm.

## Folder Structure

## Known Issues
### 1. Support for AVX512ER and AVX512PF instructions
The current version of ASpT_Refactor did not enable AVX512ER and AVX512PF instructions because the environment of the test machine does not support these instructions. You can enable this feature by modifing 
```
Makefile
```

## License