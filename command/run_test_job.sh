#!/bin/bash
source ~/../opt/intel/oneapi/setvars.sh

# download and compile
sh download_test_job.sh
sh compile_SpMM_KNL.sh

# make a directory to store the result
cd ..
result_dest=test_job_result
mkdir ${result_dest}

# remove the old result
cd data
rm SpMM_KNL_SP.out
rm SpMM_KNL_DP.out
rm SpMM_KNL_SP_preprocessing.out
rm SpMM_KNL_DP_preprocessing.out

#benchmark ASpT_SpMM_SP
echo "dataset, ASpT_GFLOPs(K=32), ASpT_GFLOPs(K=128), ASpT_diff_%(K=32+K=128)" >> SpMM_KNL_SP.out
echo "dataset, preprocessing_ratio" >> SpMM_KNL_SP_preprocessing.out

for i in `ls -d */`
do
cd ${i}
ii=$(echo "$i" | sed 's/\///g')
cd ..
# document dataset name
echo -n ${ii} >> SpMM_KNL_SP.out
echo -n "," >> SpMM_KNL_SP.out
echo -n ${ii} >> SpMM_KNL_SP_preprocessing.out
echo -n "," >> SpMM_KNL_SP_preprocessing.out
../SpMM_KNL/SpMM_ASpT_SP.x ${ii}/${ii}.mtx
echo >> SpMM_KNL_SP.out
echo >> SpMM_KNL_SP_preprocessing.out
done

#benchmark ASpT_SpMM_DP
echo "dataset, ASpT_GFLOPs(K=32), ASpT_GFLOPs(K=128), ASpT_diff_%(K=32+K=128)" >> SpMM_KNL_DP.out
echo "dataset, preprocessing_ratio" >> SpMM_KNL_DP_preprocessing.out

for i in `ls -d */`
do
cd ${i}
ii=$(echo "$i" | sed 's/\///g')
cd ..
# document dataset name
echo -n ${ii} >> SpMM_KNL_DP.out
echo -n "," >> SpMM_KNL_DP.out
echo -n ${ii} >> SpMM_KNL_DP_preprocessing.out
echo -n "," >> SpMM_KNL_DP_preprocessing.out
../SpMM_KNL/SpMM_ASpT_DP.x ${ii}/${ii}.mtx
echo >> SpMM_KNL_DP.out
echo >> SpMM_KNL_DP_preprocessing.out
done

mv *.out ../${result_dest}/
cd ..
cd command


