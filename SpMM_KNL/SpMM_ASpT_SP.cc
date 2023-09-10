// temp.grp remove

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <xmmintrin.h>
#include "mkl.h"
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <iostream>
using namespace std;

double time_in_mill_now();
double time_in_mill_now()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	double time_in_mill =
		(tv.tv_sec) * 1000.0 + (tv.tv_usec) / 1000.0;
	return time_in_mill;
}

#define ERR fprintf(stderr, "ERR\n");

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CEIL(a, b) (((a) + (b)-1) / (b))
#define FTYPE float

#define MFACTOR (32)
#define LOG_MFACTOR (5)
#define BSIZE (1024 / 1)
#define BF (BSIZE / 32)
#define INIT_GRP (10000000)
#define INIT_LIST (-1)
#define THRESHOLD (16 * 1)
#define BH (128 * 1)
#define LOG_BH (7)
#define BW (128 * 1)
#define MIN_OCC (BW * 3 / 4)
// #define MIN_OCC (-1)
#define SBSIZE (128)
#define SBF (SBSIZE / 32)
#define DBSIZE (1024)
#define DBF (DBSIZE / 32)
#define SPBSIZE (256)
#define SPBF (SPBSIZE / 32)
#define STHRESHOLD (1024 / 2 * 1)
#define SSTRIDE (STHRESHOLD / SPBF)
#define NTHREAD (68)
#define SC_SIZE (2048)

// #define SIM_VALUE

struct v_struct
{
	int row, col;
	FTYPE val;
	int grp;
};

double vari, avg;
double avg0[NTHREAD];
struct v_struct *temp_v, *gold_temp_v;
int sc, nr, nc, ne, gold_ne, npanel, mne, mne_nr;
int nr0;

int *csr_v;
int *csr_e, *csr_e0;
FTYPE *csr_ev, *csr_ev0;
// int *mcsr_v;
int *mcsr_e; // can be short type
int *mcsr_cnt;
int *mcsr_list;
int *mcsr_chk;

int *baddr, *saddr;
int num_dense;

int *special;
int *special2;
int special_p;
char scr_pad[NTHREAD][SC_SIZE];
double p_elapsed;

int compare0(const void *a, const void *b)
{
	if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0)
		return 1;
	if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0)
		return -1;
	return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}

int compare1(const void *a, const void *b)
{
	if ((((struct v_struct *)a)->row) / BH - (((struct v_struct *)b)->row) / BH > 0)
		return 1;
	if ((((struct v_struct *)a)->row) / BH - (((struct v_struct *)b)->row) / BH < 0)
		return -1;
	if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col > 0)
		return 1;
	if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col < 0)
		return -1;
	return ((struct v_struct *)a)->row - ((struct v_struct *)b)->row;
}

int compare2(const void *a, const void *b)
{
	if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0)
		return 1;
	if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0)
		return -1;
	if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp > 0)
		return 1;
	if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp < 0)
		return -1;
	return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}

void ready(int argc, char **argv)
{
	FILE *fp;
	int *loc;
	char buf[300];
	int nflag, sflag;
	int pre_count = 0, tmp_ne;
	int i;

	fprintf(stdout, "TTAAGG,%s,", argv[1]);

	////sc = atoi(argv[2]);
	sc = 128;
	// open target .mtx file
	fp = fopen(argv[1], "r");
	//scan first line
	fgets(buf, 300, fp);
	if (strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL)
		sflag = 1; // symmetric
	else
		sflag = 0;
	if (strstr(buf, "pattern") != NULL)
		nflag = 0; // non-value
	else if (strstr(buf, "complex") != NULL)
		nflag = -1;
	else
		nflag = 1;

#ifdef SYM
	sflag = 1;
#endif

	//skip comments
	while (1)
	{
		pre_count++;
		fgets(buf, 300, fp);
		if (strstr(buf, "%") == NULL)
			break;
	}
	fclose(fp);

	fp = fopen(argv[1], "r");
	for (i = 0; i < pre_count; i++)
		fgets(buf, 300, fp);

	//get matrix row, column, non-zero
	fscanf(fp, "%d %d %d", &nr, &nc, &ne);
	//document row number before using CEIL function
	nr0 = nr;
	//if set symmetric flag, double non-zero
	ne *= (sflag + 1);
	//the CEIL function is to make sure that the number of rows is a multiple of BH
	//the CEIL function always rounds up
	nr = CEIL(nr, BH) * BH;
	npanel = CEIL(nr, BH);

	//allocate memory
	temp_v = (struct v_struct *)malloc(sizeof(struct v_struct) * (ne + 1));
	gold_temp_v = (struct v_struct *)malloc(sizeof(struct v_struct) * (ne + 1));

	for (i = 0; i < ne; i++)
	{
		fscanf(fp, "%d %d", &temp_v[i].row, &temp_v[i].col);
		temp_v[i].grp = INIT_GRP;
		//this is because in matrix market format, the index starts from 1
		temp_v[i].row--;
		temp_v[i].col--;

		if (temp_v[i].row < 0 || temp_v[i].row >= nr || temp_v[i].col < 0 || temp_v[i].col >= nc)
		{
			fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].row, temp_v[i].col);
			exit(0);
		}
		//if pattern matrix, assign random value
		if (nflag == 0)
			temp_v[i].val = (FTYPE)(rand() % 1048576) / 1048576;
		else if (nflag == 1)
		{
			FTYPE ftemp;
			fscanf(fp, " %f ", &ftemp);
			temp_v[i].val = ftemp;
		}
		else
		{ // complex
			FTYPE ftemp1, ftemp2;
			fscanf(fp, " %f %f ", &ftemp1, &ftemp2);
			temp_v[i].val = ftemp1;
		}
#ifdef SIM_VALUE
		temp_v[i].val = 1.0f;
#endif
		// if symmetric, add reverse edge by swapping row and column
		if (sflag == 1)
		{
			//i++ because we need to add reverse edge
			i++;
			temp_v[i].row = temp_v[i - 1].col;
			temp_v[i].col = temp_v[i - 1].row;
			temp_v[i].val = temp_v[i - 1].val;
			temp_v[i].grp = INIT_GRP;
		}
	}
	//sort by row and column
	qsort(temp_v, ne, sizeof(struct v_struct), compare0);

	loc = (int *)malloc(sizeof(int) * (ne + 1));

	memset(loc, 0, sizeof(int) * (ne + 1));
	loc[0] = 1;
	for (i = 1; i < ne; i++)
	{
		if (temp_v[i].row == temp_v[i - 1].row && temp_v[i].col == temp_v[i - 1].col)
			loc[i] = 0;
		else
			loc[i] = 1;
	}
	for (i = 1; i <= ne; i++)
		loc[i] += loc[i - 1];
	for (i = ne; i >= 1; i--)
		loc[i] = loc[i - 1];
	loc[0] = 0;

	//segment with same row and column are merged
	for (i = 0; i < ne; i++)
	{
		temp_v[loc[i]].row = temp_v[i].row;
		temp_v[loc[i]].col = temp_v[i].col;
		temp_v[loc[i]].val = temp_v[i].val;
		temp_v[loc[i]].grp = temp_v[i].grp;
	}
	//ne is the number of non-zero with different row and column
	ne = loc[ne];
	//temp_v[ne] is the last element, which is not used
	temp_v[ne].row = nr;
	gold_ne = ne;
	//copy to gold_temp_v
	for (i = 0; i <= ne; i++)
	{
		gold_temp_v[i].row = temp_v[i].row;
		gold_temp_v[i].col = temp_v[i].col;
		gold_temp_v[i].val = temp_v[i].val;
		gold_temp_v[i].grp = temp_v[i].grp;
	}
	free(loc);

	//allocate memory for CSR format
	csr_v = (int *)malloc(sizeof(int) * (nr + 1));//csr_v is the row pointer
	csr_e0 = (int *)malloc(sizeof(int) * ne);//csr_e0 is the column index
	csr_ev0 = (FTYPE *)malloc(sizeof(FTYPE) * ne);//csr_ev0 is the value
	memset(csr_v, 0, sizeof(int) * (nr + 1));

	for (i = 0; i < ne; i++)
	{
		//csr_e0 is the column index
		csr_e0[i] = temp_v[i].col;
		//csr_ev0 is the value
		csr_ev0[i] = temp_v[i].val;
		//csr_v is the row pointer
		csr_v[1 + temp_v[i].row] = i + 1;
	}

	//this loop is to make sure that csr_v[i] is the index of the first non-zero element in row i
	for (i = 1; i < nr; i++)
	{
		if (csr_v[i] == 0)
			csr_v[i] = csr_v[i - 1];
	}
	csr_v[nr] = ne;

	csr_e = (int *)malloc(sizeof(int) * ne);
	csr_ev = (FTYPE *)malloc(sizeof(FTYPE) * ne);

	fprintf(stdout, "%d,%d,%d,", nr0, nc, ne);
}

void gen()
{
	special = (int *)malloc(sizeof(int) * ne);
	special2 = (int *)malloc(sizeof(int) * ne);
	memset(special, 0, sizeof(int) * ne);
	memset(special2, 0, sizeof(int) * ne);

	mcsr_cnt = (int *)malloc(sizeof(int) * (npanel + 1));
	mcsr_chk = (int *)malloc(sizeof(int) * (npanel + 1));
	mcsr_e = (int *)malloc(sizeof(int) * ne); // reduced later
	memset(mcsr_cnt, 0, sizeof(int) * (npanel + 1));
	memset(mcsr_chk, 0, sizeof(int) * (npanel + 1));
	memset(mcsr_e, 0, sizeof(int) * ne);

	int bv_size = CEIL(nc, 32);
	unsigned int **bv = (unsigned int **)malloc(sizeof(unsigned int *) * NTHREAD);
	for (int i = 0; i < NTHREAD; i++)
		bv[i] = (unsigned int *)malloc(sizeof(unsigned int) * bv_size);
	int **csr_e1 = (int **)malloc(sizeof(int *) * 2);
	short **coo = (short **)malloc(sizeof(short *) * 2);
	for (int i = 0; i < 2; i++)
	{
		csr_e1[i] = (int *)malloc(sizeof(int) * ne);
		coo[i] = (short *)malloc(sizeof(short) * ne);
	}

	struct timeval tt1, tt2, tt3, tt4;
	struct timeval starttime0;

	struct timeval starttime, endtime;
	gettimeofday(&starttime0, NULL);

	// filtering(WILL)
	// memcpy(csr_e1[0], csr_e0, sizeof(int)*ne);
#pragma omp parallel for num_threads(68) schedule(dynamic, 1)
	//each thread is assigned a row panel
	//nr / BH = CEIL(nr, BH) - 1 = row_panel number - 1
	for (int row_panel = 0; row_panel < nr / BH; row_panel++)
	{
		//i iterates row by row
		for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++)
		{
			//j iterates column by column
			for (int j = csr_v[i]; j < csr_v[i + 1]; j++)
			{
				//csr_e1[0][j] is the column index of the j-th non-zero element in row i
				csr_e1[0][j] = csr_e0[j];
			}
		}
	}

	gettimeofday(&starttime, NULL);

#pragma omp parallel for num_threads(68) schedule(dynamic, 1)
	for (int row_panel = 0; row_panel < nr / BH; row_panel++)
	{
		//tid is an integer ranging from 0 to 67
		int tid = omp_get_thread_num();
		int i, j, t_sum = 0;

		// coo generate and mcsr_chk
		
		memset(scr_pad[tid], 0, sizeof(char) * SC_SIZE);
		for (i = row_panel * BH; i < (row_panel + 1) * BH; i++)
		{
			//j iterates column by column
			for (j = csr_v[i]; j < csr_v[i + 1]; j++)
			{
				//(i & (BH - 1)) means current row id & (row per panel - 1), which is the row id in current panel(possibly wrong)
				coo[0][j] = (i & (BH - 1));
				//k is the column id in current panel(possibly wrong)
				int k = (csr_e0[j] & (SC_SIZE - 1));
				//if the value in scr_pad[tid][k] is smaller than THRESHOLD, then increase the value by 1
				if (scr_pad[tid][k] < THRESHOLD)
				{
					//if the value in scr_pad[tid][k] is THRESHOLD - 1, then increase t_sum by 1
					if (scr_pad[tid][k] == THRESHOLD - 1)
						t_sum++;//this means that the number of non-zero elements in current column is at least THRESHOLD
					scr_pad[tid][k]++;
				}
			}
		}

		//if the number of column with at least THRESHOLD non-zero elements is smaller than MIN_OCC, then skip
		if (t_sum < MIN_OCC)
		{
			//mcsr_chk is used to check whether the row panel is dense or not
			//0 means dense, 1 means sparse
			mcsr_chk[row_panel] = 1;
			
			mcsr_cnt[row_panel + 1] = 1;
			continue;
		}

		// sorting(merge sort)
		//🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
		//note that merge sort is STABLE, so the elements with same column index will be sorted by row index
		int flag = 0;
		//stride is the distance between two elements in the same group
		for (int stride = 1; stride <= BH / 2; stride *= 2, flag = 1 - flag)
		{
			//pivot is the starting point of each group
			for (int pivot = row_panel * BH; pivot < (row_panel + 1) * BH; pivot += stride * 2)
			{
				//i is the index of the current sorted element
				//l1, l2 mark the element to be compared and sorted
				int l1, l2;
				//there's a problem with the limit of l1 and l2: pivot + stride should not be larger than (row_panel + 1) * BH ⭐🚨
				for (i = l1 = csr_v[pivot], l2 = csr_v[pivot + stride]; l1 < csr_v[pivot + stride] && l2 < csr_v[pivot + stride * 2]; i++)
				{
					//if the column index of the l1-th element in current group is smaller than the column index of the l2-th element in current group
					if (csr_e1[flag][l1] <= csr_e1[flag][l2])
					{
						coo[1 - flag][i] = coo[flag][l1];
						csr_e1[1 - flag][i] = csr_e1[flag][l1++];
					}
					else
					{
						coo[1 - flag][i] = coo[flag][l2];
						csr_e1[1 - flag][i] = csr_e1[flag][l2++];
					}
				}
				//if l1 is smaller than csr_v[pivot + stride], then copy the rest of the elements in the first group
				while (l1 < csr_v[pivot + stride])
				{
					coo[1 - flag][i] = coo[flag][l1];
					csr_e1[1 - flag][i++] = csr_e1[flag][l1++];
				}
				//if l2 is smaller than csr_v[pivot + stride * 2], then copy the rest of the elements in the second group
				while (l2 < csr_v[pivot + stride * 2])
				{
					coo[1 - flag][i] = coo[flag][l2];
					csr_e1[1 - flag][i++] = csr_e1[flag][l2++];
				}
				//only one of these two while loops will be executed one time
			}
		}
		//the flag here is used to indicate which array is the final sorted array

		int weight = 1;

		int cq = 0, cr = 0;

		// dense bit extract (and mcsr_e making)
		//iterate row by row in current panel
		for (i = csr_v[row_panel * BH] + 1; i < csr_v[(row_panel + 1) * BH]; i++)
		{
			//find the number of non-zero elements in current column
			if (csr_e1[flag][i - 1] == csr_e1[flag][i])
				weight++;
			else
			{
				if (weight >= THRESHOLD)
				{
					cr++;
				} // if(cr == BW) { cq++; cr=0;}
				weight = 1;
			}
		}
		// int reminder = (csr_e1[flag][i-1]&31);
		// check if the number of non-zero elements in the last column is larger than THRESHOLD
		if (weight >= THRESHOLD)
		{
			cr++;
		} // if(cr == BW) { cq++; cr=0; }
		// TODO = occ control
		//mcsr_cnt is the number of tiles in current row panel, + 1 because all the sparse columns are stored in the last tile
		mcsr_cnt[row_panel + 1] = CEIL(cr, BW) + 1;
	}

	////gettimeofday(&tt1, NULL);
	// prefix-sum
	for (int i = 1; i <= npanel; i++)
		mcsr_cnt[i] += mcsr_cnt[i - 1];
	// mcsr_e[0] = 0;
	mcsr_e[BH * mcsr_cnt[npanel]] = ne;

	////gettimeofday(&tt2, NULL);

#pragma omp parallel for num_threads(68) schedule(dynamic, 1)
	for (int row_panel = 0; row_panel < nr / BH; row_panel++)
	{
		int tid = omp_get_thread_num();
		//if currnet row panel is dense
		if (mcsr_chk[row_panel] == 0)
		{
			int i, j;
			int flag = 0;
			int cq = 0, cr = 0;
			for (int stride = 1; stride <= BH / 2; stride *= 2, flag = 1 - flag)
				;// this for loop is used to find the stride, as well as the flag--which array is the final sorted array
			//base is the starting 2D tile row index of the current tile
			int base = (mcsr_cnt[row_panel] * BH);
			//mfactor is the number of tiles in current row panel
			int mfactor = mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel];
			int weight = 1;

			// mcsr_e making
			//i iterates all the non-zero elements in current row panel
			for (i = csr_v[row_panel * BH] + 1; i < csr_v[(row_panel + 1) * BH]; i++)
			{
				if (csr_e1[flag][i - 1] == csr_e1[flag][i])
					weight++;
				else
				{
					//reminder is the column index of the last non-zero element in current column % 32
					int reminder = (csr_e1[flag][i - 1] & 31);
					if (weight >= THRESHOLD)
					{
						cr++;
						//bv is the bit vector, which is used to mark whether the column index is in the sparse tile or not
						//csr_e1[flag][i-1] shifts right by 5 bits, which is the same as dividing by 32
						bv[tid][csr_e1[flag][i - 1] >> 5] |= (1 << reminder);
						//j iterates all the non-zero elements in current column
						for (j = i - weight; j <= i - 1; j++)
						{
							//now mcsr_e documents the number of non-zero elements in each row of each tile
							//now only elements in dense columns are counted
							mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
						}
					}
					else
					{
						// bv[tid][csr_e1[flag][i-1]>>5] &= (~0 - (1<<reminder));
						bv[tid][csr_e1[flag][i - 1] >> 5] &= (0xFFFFFFFF - (1 << reminder));
					}

					//if the number of non-zero elements in current column is larger than BW, then move to the next tile
					//BW is the number of columns in each tile
					if (cr == BW)
					{
						cq++;
						cr = 0;
					}
					weight = 1;
				}
			}

			// fprintf(stderr, "inter : %d\n", i);

			//process the last elements in current column
			int reminder = (csr_e1[flag][i - 1] & 31);
			if (weight >= THRESHOLD)
			{
				cr++;
				bv[tid][csr_e1[flag][i - 1] >> 5] |= (1 << reminder);
				for (j = i - weight; j <= i - 1; j++)
				{
					mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
				}
			}
			else
			{
				bv[tid][csr_e1[flag][i - 1] >> 5] &= (0xFFFFFFFF - (1 << reminder));
			}
			// reordering
			//delta is the number of tiles in current row panel
			int delta = mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel];
			//base0 is the starting 2D tile row index of the current row panel
			int base0 = mcsr_cnt[row_panel] * BH;
			//i iterate all the rows in current row panel
			for (i = row_panel * BH; i < (row_panel + 1) * BH; i++)
			{
				//base is the starting 2D tile index of the current row
				int base = base0 + (i - row_panel * BH) * delta;
				//dpnt points to the first non-zero element in current row, which is the starting point of first dense tile
				// mcsr_e[base] = csr_v[i] = number of elements before current row in current row panel
				int dpnt = mcsr_e[base] = csr_v[i];
				//j iterates all tiles in current row
				for (int j = 1; j < delta; j++)
				{
					//partial prefix sum
					//making mcsr_e a complete 2D tile row pointer
					mcsr_e[base + j] += mcsr_e[base + j - 1];
				}
				//spnt is the starting point of the sparse tile
				int spnt = mcsr_e[mcsr_cnt[row_panel] * BH + (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) * (i - row_panel * BH + 1) - 1];

				//csr_v[i+1] is the start point of the next row
				//csr_v[i+1] - spnt gives the number of non-zero elements in the sparse tile in current row
				avg0[tid] += csr_v[i + 1] - spnt;
				//j iterates all the non-zero elements in current row
				for (j = csr_v[i]; j < csr_v[i + 1]; j++)
				{
					//k is the column index of the j-th non-zero element in current row
					int k = csr_e0[j];
					if ((bv[tid][k >> 5] & (1 << (k & 31))))
					{
						//sort the element in dense tile
						csr_e[dpnt] = csr_e0[j];
						csr_ev[dpnt++] = csr_ev0[j];
					}
					else
					{
						//sort the element in sparse tile
						csr_e[spnt] = csr_e0[j];
						csr_ev[spnt++] = csr_ev0[j];
					}
				}
			}
		}
		else
		{
			int base0 = mcsr_cnt[row_panel] * BH;
			memcpy(&mcsr_e[base0], &csr_v[row_panel * BH], sizeof(int) * BH);
			avg0[tid] += csr_v[(row_panel + 1) * BH] - csr_v[row_panel * BH];
			int bidx = csr_v[row_panel * BH];
			int bseg = csr_v[(row_panel + 1) * BH] - bidx;
			//sparse row panel's non-zero elements are stored in CSR format
			memcpy(&csr_e[bidx], &csr_e0[bidx], sizeof(int) * bseg);
			memcpy(&csr_ev[bidx], &csr_ev0[bidx], sizeof(FTYPE) * bseg);
		}
	}

	for (int i = 0; i < NTHREAD; i++)
		avg += avg0[i];
	avg /= (double)nr;
	//avg gives the average number of sparse tiles' non-zero elements in each row

	////gettimeofday(&tt3, NULL);

	for (int i = 0; i < nr; i++)
	{
		//idx is the index of next row in mcsr_e
		int idx = (mcsr_cnt[i >> LOG_BH]) * BH + (mcsr_cnt[(i >> LOG_BH) + 1] - mcsr_cnt[i >> LOG_BH]) * ((i & (BH - 1)) + 1);
		//diff gives the number of sparse tiles' non-zero elements in current row
		int diff = csr_v[i + 1] - mcsr_e[idx - 1];
		//r is the difference between the number of sparse tiles' non-zero elements in current row and the average number of sparse tiles' non-zero elements in each row
		double r = ((double)diff - avg);
		//vari is the variance of the number of sparse tiles' non-zero elements in each row
		vari += r * r;

		//if the number of sparse tiles' non-zero elements in current row is larger than STHRESHOLD, then split the row
		if (diff >= STHRESHOLD)
		{
			int pp = (diff) / STHRESHOLD;
			for (int j = 0; j < pp; j++)
			{
				//special is used to store the row id of the split row
				special[special_p] = i;
				//special2 is used to store the column id(of the first non-zero element in the split row) of the split row
				special2[special_p] = j * STHRESHOLD;
				special_p++;
			}
		}
	}
	vari /= (double)nr;

	gettimeofday(&endtime, NULL);

	double elapsed0 = ((starttime.tv_sec - starttime0.tv_sec) * 1000000 + starttime.tv_usec - starttime0.tv_usec) / 1000000.0;
	// double elapsed1 = ((tt1.tv_sec-starttime.tv_sec)*1000000 + tt1.tv_usec-starttime.tv_usec)/1000000.0;
	// double elapsed2 = ((tt2.tv_sec-tt1.tv_sec)*1000000 + tt2.tv_usec-tt1.tv_usec)/1000000.0;
	// double elapsed3 = ((tt3.tv_sec-tt2.tv_sec)*1000000 + tt3.tv_usec-tt2.tv_usec)/1000000.0;
	// double elapsed4 = ((endtime.tv_sec-tt3.tv_sec)*1000000 + endtime.tv_usec-tt3.tv_usec)/1000000.0;
	// fprintf(stdout, "(%f %f %f %f %f)", elapsed0*1000, elapsed1*1000, elapsed2*1000, elapsed3*1000, elapsed4*1000);
	//process elapsed time
	p_elapsed = ((endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec) / 1000000.0;
	fprintf(stdout, "%f,%f,", elapsed0 * 1000, p_elapsed * 1000);

	for (int i = 0; i < NTHREAD; i++)
		free(bv[i]);
	for (int i = 0; i < 2; i++)
	{
		free(csr_e1[i]);
		free(coo[i]);
	}
	free(bv);
	free(csr_e1);
	free(coo);
}

void mprocess()
{
	FILE *fpo = fopen("SpMM_KNL_SP.out", "a");
	FILE *fpo2 = fopen("SpMM_KNL_SP_preprocessing.out", "a");

	double elapsed[3];
	FTYPE *vin, *vout;
	FTYPE *vout_gold;
	vin = (FTYPE *)_mm_malloc(sizeof(FTYPE) * nc * sc, 64);
	vout = (FTYPE *)_mm_malloc(sizeof(FTYPE) * nr * sc, 64);

	__assume_aligned(csr_v, 64);
	__assume_aligned(csr_e, 64);
	__assume_aligned(csr_ev, 64);
	//__assume_aligned(mcsr_cnt, 64);
	//__assume_aligned(mcsr_e, 64);
	//__assume_aligned(mcsr_list, 64);
	__assume_aligned(vin, 64);
	__assume_aligned(vout, 64);

	//sc is the column number of the dense matrix
	for (sc = 8; sc <= 128; sc *= 4)
	{

		struct timeval starttime, endtime;

		//initialize vin and vout
		//vout is the output matrix(in the form of 1D array)
		memset(vout, 0, sizeof(FTYPE) * nr * sc);
#pragma vector aligned
#pragma omp parallel for num_threads(68)
		for (int i = 0; i < nc * sc; i++)
		{
			//vin is the dense matrix(in the form of 1D array)
			vin[i] = (FTYPE)(rand() % 1048576) / 1048576;
#ifdef SIM_VALUE
			vin[i] = 1;
#endif
		}

		// double tot_time;
// cout << "v" << ne/nc << endl;
#define ITER (128 * 1 / 128)
		if (vari < 5000 * 1 / 1 * 1)
		{

			gettimeofday(&starttime, NULL);
			////begin
			//calculate the output matrix ITER times and add it to vout
			for (int loop = 0; loop < ITER; loop++)
			{
#pragma ivdep //ivdep is a hint to the compiler that the loop is safe for instruction-level parallelism
#pragma vector aligned //vector aligned is a hint to the compiler that the loop is safe for vectorization
#pragma temporal(vin) //temporal is a hint to the compiler that the loop is safe for temporal locality
#pragma omp parallel for num_threads(136) schedule(dynamic, 1)
				for (int row_panel = 0; row_panel < nr / BH; row_panel++)
				{
					// dense
					int stride;
					//stride iterates all the dense tiles in current row panel
					for (stride = 0; stride < mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel] - 1; stride++)
					{
						
						//i iterates all the rows in current dense tile, which is also the row id in sparse matrix
						for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++)
						{
							//dummy is the tile_row_ptr	index of the first non-zero element in current dense tile in current row
							int dummy = mcsr_cnt[row_panel] * BH + (i & (BH - 1)) * (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) + stride;
							//loc1 is the column index of the first non-zero element in current dense tile in current row
							//loc2 is the column index of the last non-zero element in current dense tile in current row + 1
							int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy + 1];

							//interm is set to be loc1 + (((loc2 - loc1) >> 3) << 3), which is the largest multiple of 8 that is smaller than (loc2 - loc1)
							int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
							int j;
							//the two loops below are used to iterate all the non-zero elements in current dense tile in current row and calculate corresponding elements in output matrix
							for (j = loc1; j < interm; j += 8)
							{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vin : _MM_HINT_T1 //prefetch is a hint to the compiler that the loop is safe for spatial locality
#pragma temporal(vin) //temporal is a hint to the compiler that the loop is safe for temporal locality
								
								for (int k = 0; k < sc; k++)
								{
									vout[i * sc + k] = vout[i * sc + k] + csr_ev[j] * vin[csr_e[j] * sc + k] + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k] + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k] + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k] + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k] + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k] + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k] + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
								}
							}
							for (; j < loc2; j++)
							{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vout : _MM_HINT_T1
#pragma temporal(vout)
								for (int k = 0; k < sc; k++)
								{
									vout[i * sc + k] += csr_ev[j] * vin[csr_e[j] * sc + k];
								}
							}
						}
					}
					// sparse
					for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++)
					{
						//the sparse part is stored in CSR format
						//now stride is the index of the sparse tile in current row panel
						//dummy here is the tile_row_ptr index of the first non-zero element in current sparse tile in current row
						int dummy = mcsr_cnt[row_panel] * BH + (i & (BH - 1)) * (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) + stride;
						int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy + 1];

						// printf("(%d %d %d %d %d)\n", i, csr_v[i], loc1, csr_v[i+1], loc2);
						// printf("%d %d %d %d %d %d %d\n", i, dummy, stride, csr_v[i], loc1, csr_v[i+1], loc2);

						int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
						int j;
						for (j = loc1; j < interm; j += 8)
						{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vin : _MM_HINT_T1
#pragma temporal(vin)
							for (int k = 0; k < sc; k++)
							{
								vout[i * sc + k] = vout[i * sc + k] + csr_ev[j] * vin[csr_e[j] * sc + k] + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k] + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k] + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k] + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k] + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k] + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k] + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
							}
						}
						for (; j < loc2; j++)
						{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vout : _MM_HINT_T1
#pragma temporal(vout)
							for (int k = 0; k < sc; k++)
							{
								vout[i * sc + k] += csr_ev[j] * vin[csr_e[j] * sc + k];
							}
						}
					}
				}
				////end
			}
			gettimeofday(&endtime, NULL);
		}
		else
		{ // big var

			gettimeofday(&starttime, NULL);
			////begin
			for (int loop = 0; loop < ITER; loop++)
			{
#pragma ivdep
#pragma vector aligned
#pragma temporal(vin)
#pragma omp parallel for num_threads(136) schedule(dynamic, 1)
				for (int row_panel = 0; row_panel < nr / BH; row_panel++)
				{
					// dense
					int stride;
					for (stride = 0; stride < mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel] - 1; stride++)
					{

						for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++)
						{
							int dummy = mcsr_cnt[row_panel] * BH + (i & (BH - 1)) * (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) + stride;
							int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy + 1];

							int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
							int j;
							for (j = loc1; j < interm; j += 8)
							{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vin : _MM_HINT_T1
#pragma temporal(vin)
								for (int k = 0; k < sc; k++)
								{
									vout[i * sc + k] = vout[i * sc + k] + csr_ev[j] * vin[csr_e[j] * sc + k] + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k] + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k] + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k] + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k] + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k] + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k] + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
								}
							}
							for (; j < loc2; j++)
							{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vout : _MM_HINT_T1
#pragma temporal(vout)
								for (int k = 0; k < sc; k++)
								{
									vout[i * sc + k] += csr_ev[j] * vin[csr_e[j] * sc + k];
								}
							}
						}
					}
					// sparse
					for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++)
					{

						int dummy = mcsr_cnt[row_panel] * BH + (i & (BH - 1)) * (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) + stride;
						int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy + 1];

						loc1 += ((loc2 - loc1) / STHRESHOLD) * STHRESHOLD;

						int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
						int j;
						for (j = loc1; j < interm; j += 8)
						{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vin : _MM_HINT_T1
#pragma temporal(vin)
							for (int k = 0; k < sc; k++)
							{
								vout[i * sc + k] = vout[i * sc + k] + csr_ev[j] * vin[csr_e[j] * sc + k] + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k] + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k] + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k] + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k] + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k] + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k] + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
							}
						}
						for (; j < loc2; j++)
						{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vout : _MM_HINT_T1
#pragma temporal(vout)
							for (int k = 0; k < sc; k++)
							{
								vout[i * sc + k] += csr_ev[j] * vin[csr_e[j] * sc + k];
							}
						}
					}
				}
#pragma ivdep
#pragma vector aligned
#pragma temporal(vin)
#pragma omp parallel for num_threads(136) schedule(dynamic, 1)
				for (int row_panel = 0; row_panel < special_p; row_panel++)
				{
					int i = special[row_panel];

					int dummy = mcsr_cnt[i >> LOG_BH] * BH + ((i & (BH - 1)) + 1) * (mcsr_cnt[(i >> LOG_BH) + 1] - mcsr_cnt[i >> LOG_BH]);

					int loc1 = mcsr_e[dummy - 1] + special2[row_panel];
					int loc2 = loc1 + STHRESHOLD;

					// int interm = loc1 + (((loc2 - loc1)>>3)<<3);
					int j;
					// assume to 128
					FTYPE temp_r[128] = {
						0,
					};
					// for(int e=0;e<128;e++) {
					//	temp_r[e] = 0.0f;
					// }

					for (j = loc1; j < loc2; j += 8)
					{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vin : _MM_HINT_T1
#pragma temporal(vin)
						for (int k = 0; k < sc; k++)
						{
							temp_r[k] = temp_r[k] + csr_ev[j] * vin[csr_e[j] * sc + k] + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k] + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k] + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k] + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k] + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k] + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k] + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
						}
					}
#pragma ivdep
					for (int k = 0; k < sc; k++)
					{
#pragma omp atomic
						vout[i * sc + k] += temp_r[k];
					}
				}

			} // end loop
			gettimeofday(&endtime, NULL);
		}

		// double elapsed = ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
		if (sc == 8)
			elapsed[0] = ((endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec) / 1000000.0;
		else if (sc == 32)
			elapsed[1] = ((endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec) / 1000000.0;
		else
			elapsed[2] = ((endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec) / 1000000.0;
	}
	sc = 128;
	vout_gold = (FTYPE *)malloc(sizeof(FTYPE) * nr * sc);
	memset(vout_gold, 0, sizeof(FTYPE) * nr * sc);

#define VALIDATE
#if defined VALIDATE
	// validate
	for (int i = 0; i < nr * sc; i++)
	{
		vout_gold[i] = 0.0f;
	}
	for (sc = 128; sc <= 128; sc *= 4)
	{
		for (int i = 0; i < gold_ne; i++)
		{
			for (int j = 0; j < sc; j++)
			{
				//vout_gold stored the correct output matrix
				vout_gold[gold_temp_v[i].row * sc + j] += vin[sc * gold_temp_v[i].col + j] * gold_temp_v[i].val;
			}
		}
	}
	sc = 128;
	int num_diff = 0;
	//the loop iterates all the elements in the output matrix
	for (int i = 0; i < nr * sc; i++)
	{
		//vout_gold is multiplied by ITER, because the output matrix is calculated ITER times
		FTYPE p1 = vout_gold[i] * ITER;
		FTYPE p2 = vout[i];

		//find absolute value
		if (p1 < 0)
			p1 *= -1;
		if (p2 < 0)
			p2 *= -1;
		FTYPE diff;
		diff = p1 - p2;
		if (diff < 0)
			diff *= -1;
		//diff = ||p1|-|p2||
		//if diff is larger than 1% of the larger one of p1 and p2, then the two elements are different
		//scale is used to avoid the situation that p1 and p2 are both very small
		if (diff / MAX(p1, p2) > 0.01)
		{
			// if(num_diff < 20*1*1) fprintf(stdout, "%d %f %f\n", i, vout[i], vout_gold[i]*ITER);
			// if(vout[i] < vout_gold[i]) fprintf(stdout, "%d %f %f\n", i, vout[i], vout_gold[i]);

			num_diff++;
		}
	}
	//      fprintf(stdout, "num_diff : %d\n", num_diff);
	//print the percentage of different elements
	fprintf(stdout, "%f,", (double)num_diff / (nr * sc) * 100);
//      fprintf(stdout, "ne : %d\n", gold_ne);
#endif
	// fprintf(stdout, "%f,%f\n", (double)elapsed*1000/ITER, (double)ne*2*sc*ITER/elapsed/1000000000);
	fprintf(stdout, "%f,", (double)ne * 2 * 8 * ITER / elapsed[0] / 1000000000);
	fprintf(stdout, "%f,%f,", (double)ne * 2 * 32 * ITER / elapsed[1] / 1000000000, (double)ne * 2 * 128 * ITER / elapsed[2] / 1000000000);
	//the ratio of the time of preprocessing to the time of calculating the output matrix
	fprintf(stdout, "%f\n", p_elapsed / (elapsed[2] / ITER));

	//ASpT GFLOPS(K=32,K=128)
	fprintf(fpo, "%f,%f,", (double)ne * 2 * 32 * ITER / elapsed[1] / 1000000000, (double)ne * 2 * 128 * ITER / elapsed[2] / 1000000000);
	//num_diff (K=32+K=128)
	fprintf(fpo, "%f,", (double)num_diff / (nr * sc) * 100);
	fprintf(fpo2, "%f", p_elapsed / (elapsed[2] / ITER));
	fclose(fpo);
	fclose(fpo2);
}

int main(int argc, char **argv)
{
	ready(argc, argv);
	gen();
	// gen_structure();
	mprocess();
}
