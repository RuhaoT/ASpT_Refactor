# ASpT_Refactor
## Introduction
This is a repository for the refactor of ASpT.
You can found the original ASpT repository [here](http://gitlab.hpcrl.cse.ohio-state.edu/chong/ppopp19_ae)
This repository is still under construction.

## Installation and Usage
### Install ASpT_Refactor
```shell
git clone https://github.com/RuhaoT/ASpT_Refactor.git
```

### Test Job: Verify web-Google and web-Stanford matrix
This test job is an functional verification of ASpT SpMM algorithm.
It process two matrices: web-Google and web-Stanford, and multiply them with a random matrix using ASpT SpMM algorithm(both single precision and double precision).

Software Requirements:
```
Intel ICC compiler
```
After the compiler is ready, modify the following files based on the location of the compiler:
```
misc/Makefile.in --> line 1 & 2
command/run_test_job.sh --> line 2
```

To run this test job:
```shell
cd command
sh ./run_test_job.sh
```
The results are stored in `test_job_result` folder.

## Folder Structure
This section is work in progress.

## Known Issues
### 1. AVX512ER and AVX512PF instructions are disabled
The AVX512ER and AVX512PF instructions are mannually disabled in the Makefile.
To Enable these instructions, please add the following lines to each Makefile:
```makefile
CFLAG += -MIC-AVX512
```

## License
This section is work in progress.