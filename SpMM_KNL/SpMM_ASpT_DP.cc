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
#define FTYPE double

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

double vari_per_row, avg_sparse_elem_in_row;
double avg0[NTHREAD];
struct v_struct *temp_matrix_vector, *validate_temp_matrix_vector;
int num_dense_matrix_column, number_row, number_column, number_nonzero, validate_num_element, number_row_panel, mne, mne_nr;
int original_number_row;

int *csr_row_ptr;
int *ASpT_csr_column_index, *csr_column_index;
FTYPE *ASpT_csr_values, *csr_values;
// int *mcsr_v;
int *ASpT_tile_row_ptr; // can be short type
int *total_tile_number;
int *mcsr_list;
int *is_row_panel_sparse;

int *baddr, *saddr;
int num_dense;

int *special_row_id;
int *special_col_id;
int special_p;
char number_elements_in_column[NTHREAD][SC_SIZE];
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

void load_matrix_into_CSR(int argc, char **argv)
{
	FILE *sparse_matrix_file;
	int *loc;
	char buf[300];
	int nonzero_element_type, is_symmetric;
	int lines_to_be_skiped = 0, tmp_ne;
	int i;

	fprintf(stdout, "TTAAGG,%s,", argv[1]);

	////num_dense_matrix_column = atoi(argv[2]);
	num_dense_matrix_column = 128;
	// open target .mtx file
	sparse_matrix_file = fopen(argv[1], "r");
	//scan first line
	fgets(buf, 300, sparse_matrix_file);
	if (strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL)
		is_symmetric = 1; // symmetric
	else
		is_symmetric = 0;
	if (strstr(buf, "pattern") != NULL)
		nonzero_element_type = 0; // non-value
	else if (strstr(buf, "complex") != NULL)
		nonzero_element_type = -1;
	else
		nonzero_element_type = 1;

#ifdef SYM
	is_symmetric = 1;
#endif

	//skip comments
	while (1)
	{
		lines_to_be_skiped++;
		fgets(buf, 300, sparse_matrix_file);
		if (strstr(buf, "%") == NULL)
			break;
	}
	fclose(sparse_matrix_file);

	sparse_matrix_file = fopen(argv[1], "r");
	for (i = 0; i < lines_to_be_skiped; i++)
		fgets(buf, 300, sparse_matrix_file);

	//get matrix row, column, non-zero
	fscanf(sparse_matrix_file, "%d %d %d", &number_row, &number_column, &number_nonzero);
	//document row number before using CEIL function
	original_number_row = number_row;
	//if set symmetric flag, double non-zero
	number_nonzero *= (is_symmetric + 1);
	//the CEIL function is to make sure that the number of rows is a multiple of BH
	//the CEIL function always rounds up
	number_row = CEIL(number_row, BH) * BH;
	number_row_panel = CEIL(number_row, BH);

	//allocate memory
	temp_matrix_vector = (struct v_struct *)malloc(sizeof(struct v_struct) * (number_nonzero + 1));
	//the validate_temp_matrix_vector is used to validate the result
	validate_temp_matrix_vector = (struct v_struct *)malloc(sizeof(struct v_struct) * (number_nonzero + 1));

	for (i = 0; i < number_nonzero; i++)
	{
		fscanf(sparse_matrix_file, "%d %d", &temp_matrix_vector[i].row, &temp_matrix_vector[i].col);
		temp_matrix_vector[i].grp = INIT_GRP;
		//this is because in matrix market format, the index starts from 1
		temp_matrix_vector[i].row--;
		temp_matrix_vector[i].col--;

		if (temp_matrix_vector[i].row < 0 || temp_matrix_vector[i].row >= number_row || temp_matrix_vector[i].col < 0 || temp_matrix_vector[i].col >= number_column)
		{
			fprintf(stdout, "A vertex id is out of range %d %d\n", temp_matrix_vector[i].row, temp_matrix_vector[i].col);
			exit(0);
		}
		//if pattern matrix, assign random value
		if (nonzero_element_type == 0)
			temp_matrix_vector[i].val = (FTYPE)(rand() % 1048576) / 1048576;
		else if (nonzero_element_type == 1)
		{
			FTYPE ftemp;
			fscanf(sparse_matrix_file, " %f ", &ftemp);
			temp_matrix_vector[i].val = ftemp;
		}
		else
		{ // complex
			FTYPE ftemp1, ftemp2;
			fscanf(sparse_matrix_file, " %f %f ", &ftemp1, &ftemp2);
			temp_matrix_vector[i].val = ftemp1;
		}
#ifdef SIM_VALUE
		temp_matrix_vector[i].val = 1.0f;
#endif
		// if symmetric, add reverse edge by swapping row and column
		if (is_symmetric == 1)
		{
			//i++ because we need to add reverse edge
			i++;
			temp_matrix_vector[i].row = temp_matrix_vector[i - 1].col;
			temp_matrix_vector[i].col = temp_matrix_vector[i - 1].row;
			temp_matrix_vector[i].val = temp_matrix_vector[i - 1].val;
			temp_matrix_vector[i].grp = INIT_GRP;
		}
	}
	//sort by row and column
	qsort(temp_matrix_vector, number_nonzero, sizeof(struct v_struct), compare0);

	loc = (int *)malloc(sizeof(int) * (number_nonzero + 1));

	memset(loc, 0, sizeof(int) * (number_nonzero + 1));
	loc[0] = 1;
	for (i = 1; i < number_nonzero; i++)
	{
		if (temp_matrix_vector[i].row == temp_matrix_vector[i - 1].row && temp_matrix_vector[i].col == temp_matrix_vector[i - 1].col)
			loc[i] = 0;
		else
			loc[i] = 1;
	}
	for (i = 1; i <= number_nonzero; i++)
		loc[i] += loc[i - 1];
	for (i = number_nonzero; i >= 1; i--)
		loc[i] = loc[i - 1];
	loc[0] = 0;

	//segment with same row and column are merged
	for (i = 0; i < number_nonzero; i++)
	{
		temp_matrix_vector[loc[i]].row = temp_matrix_vector[i].row;
		temp_matrix_vector[loc[i]].col = temp_matrix_vector[i].col;
		temp_matrix_vector[loc[i]].val = temp_matrix_vector[i].val;
		temp_matrix_vector[loc[i]].grp = temp_matrix_vector[i].grp;
	}
	//number_nonzero is the number of non-zero with different row and column
	number_nonzero = loc[number_nonzero];
	//temp_matrix_vector[number_nonzero] is the last element, which is not used
	temp_matrix_vector[number_nonzero].row = number_row;
	validate_num_element = number_nonzero;
	//copy to validate_temp_matrix_vector
	for (i = 0; i <= number_nonzero; i++)
	{
		validate_temp_matrix_vector[i].row = temp_matrix_vector[i].row;
		validate_temp_matrix_vector[i].col = temp_matrix_vector[i].col;
		validate_temp_matrix_vector[i].val = temp_matrix_vector[i].val;
		validate_temp_matrix_vector[i].grp = temp_matrix_vector[i].grp;
	}
	free(loc);

	//allocate memory for CSR format
	csr_row_ptr = (int *)malloc(sizeof(int) * (number_row + 1));//csr_row_ptr is the row pointer
	csr_column_index = (int *)malloc(sizeof(int) * number_nonzero);//csr_column_index is the column index
	csr_values = (FTYPE *)malloc(sizeof(FTYPE) * number_nonzero);//csr_values is the value
	memset(csr_row_ptr, 0, sizeof(int) * (number_row + 1));

	for (i = 0; i < number_nonzero; i++)
	{
		//csr_column_index is the column index
		csr_column_index[i] = temp_matrix_vector[i].col;
		//csr_values is the value
		csr_values[i] = temp_matrix_vector[i].val;
		//csr_row_ptr is the row pointer
		csr_row_ptr[1 + temp_matrix_vector[i].row] = i + 1;
	}

	//this loop is to make sure that csr_row_ptr[i] is the index of the first non-zero element in row i
	for (i = 1; i < number_row; i++)
	{
		if (csr_row_ptr[i] == 0)
			csr_row_ptr[i] = csr_row_ptr[i - 1];
	}
	csr_row_ptr[number_row] = number_nonzero;

	ASpT_csr_column_index = (int *)malloc(sizeof(int) * number_nonzero);
	ASpT_csr_values = (FTYPE *)malloc(sizeof(FTYPE) * number_nonzero);

	fprintf(stdout, "%d,%d,%d,", original_number_row, number_column, number_nonzero);
}

void ASpT_preprocess()
{
	special_row_id = (int *)malloc(sizeof(int) * number_nonzero);
	special_col_id = (int *)malloc(sizeof(int) * number_nonzero);
	memset(special_row_id, 0, sizeof(int) * number_nonzero);
	memset(special_col_id, 0, sizeof(int) * number_nonzero);

	total_tile_number = (int *)malloc(sizeof(int) * (number_row_panel + 1));
	is_row_panel_sparse = (int *)malloc(sizeof(int) * (number_row_panel + 1));
	ASpT_tile_row_ptr = (int *)malloc(sizeof(int) * number_nonzero); // reduced later
	memset(total_tile_number, 0, sizeof(int) * (number_row_panel + 1));
	memset(is_row_panel_sparse, 0, sizeof(int) * (number_row_panel + 1));
	memset(ASpT_tile_row_ptr, 0, sizeof(int) * number_nonzero);

	int bv_size = CEIL(number_column, 32);
	unsigned int **bit_vector = (unsigned int **)malloc(sizeof(unsigned int *) * NTHREAD);
	for (int i = 0; i < NTHREAD; i++)
		bit_vector[i] = (unsigned int *)malloc(sizeof(unsigned int) * bv_size);
	int **csr_column_index_sorted = (int **)malloc(sizeof(int *) * 2);
	short **coo = (short **)malloc(sizeof(short *) * 2);
	for (int i = 0; i < 2; i++)
	{
		csr_column_index_sorted[i] = (int *)malloc(sizeof(int) * number_nonzero);
		coo[i] = (short *)malloc(sizeof(short) * number_nonzero);
	}

	struct timeval tt1, tt2, tt3, tt4;
	struct timeval starttime0;

	struct timeval start_time, endtime;
	gettimeofday(&starttime0, NULL);

	// filtering(WILL)
	// memcpy(csr_column_index_sorted[0], csr_column_index, sizeof(int)*ne);
#pragma omp parallel for num_threads(68) schedule(dynamic, 1)
	//each thread is assigned a row panel
	//number_row / BH = CEIL(number_row, BH) - 1 = row_panel number - 1
	for (int row_panel = 0; row_panel < number_row / BH; row_panel++)
	{
		//i iterates row by row
		for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++)
		{
			//j iterates column by column
			for (int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
			{
				//csr_column_index_sorted[0][j] is the column index of the j-th non-zero element in row i
				csr_column_index_sorted[0][j] = csr_column_index[j];
			}
		}
	}

	gettimeofday(&start_time, NULL);

#pragma omp parallel for num_threads(68) schedule(dynamic, 1)
	for (int row_panel = 0; row_panel < number_row / BH; row_panel++)
	{
		//tid is an integer ranging from 0 to 67
		int tid = omp_get_thread_num();
		int i, j, num_dense_column = 0;

		// coo generate and is_row_panel_sparse
		
		memset(number_elements_in_column[tid], 0, sizeof(char) * SC_SIZE);
		for (i = row_panel * BH; i < (row_panel + 1) * BH; i++)
		{
			//j iterates column by column
			for (j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
			{
				//(i & (BH - 1)) means current row id & (row per panel - 1), which is the row id in current panel(possibly wrong)
				coo[0][j] = (i & (BH - 1));
				//k is the column id in current panel(possibly wrong)
				int k = (csr_column_index[j] & (SC_SIZE - 1));
				//if the value in number_elements_in_column[tid][k] is smaller than THRESHOLD, then increase the value by 1
				if (number_elements_in_column[tid][k] < THRESHOLD)
				{
					//if the value in number_elements_in_column[tid][k] is THRESHOLD - 1, then increase num_dense_column by 1
					if (number_elements_in_column[tid][k] == THRESHOLD - 1)
						num_dense_column++;//this means that the number of non-zero elements in current column is at least THRESHOLD
					number_elements_in_column[tid][k]++;
				}
			}
		}

		//if the number of column with at least THRESHOLD non-zero elements is smaller than MIN_OCC, then skip
		if (num_dense_column < MIN_OCC)
		{
			//is_row_panel_sparse is used to check whether the row panel is dense or not
			//0 means dense, 1 means sparse
			is_row_panel_sparse[row_panel] = 1;
			
			total_tile_number[row_panel + 1] = 1;
			continue;
		}

		// sorting(merge sort)
		//🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
		//note that merge sort is STABLE, so the elements with same column index will be sorted by row index
		int flag = 0;
		for (int stride = 1; stride <= BH / 2; stride *= 2, flag = 1 - flag)
		{
			//pivot is the starting point of each group
			for (int pivot = row_panel * BH; pivot < (row_panel + 1) * BH; pivot += stride * 2)
			{
				//i is the index of the current sorted element
				//l1, l2 mark the element to be compared and sorted
				int l1, l2;
				//there's a problem with the limit of l1 and l2: pivot + stride should not be larger than (row_panel + 1) * BH ⭐🚨
				for (i = l1 = csr_row_ptr[pivot], l2 = csr_row_ptr[pivot + stride]; l1 < csr_row_ptr[pivot + stride] && l2 < csr_row_ptr[pivot + stride * 2]; i++)
				{
					//if the column index of the l1-th element in current group is smaller than the column index of the l2-th element in current group
					if (csr_column_index_sorted[flag][l1] <= csr_column_index_sorted[flag][l2])
					{
						coo[1 - flag][i] = coo[flag][l1];
						csr_column_index_sorted[1 - flag][i] = csr_column_index_sorted[flag][l1++];
					}
					else
					{
						coo[1 - flag][i] = coo[flag][l2];
						csr_column_index_sorted[1 - flag][i] = csr_column_index_sorted[flag][l2++];
					}
				}
				//if l1 is smaller than csr_row_ptr[pivot + stride], then copy the rest of the elements in the first group
				while (l1 < csr_row_ptr[pivot + stride])
				{
					coo[1 - flag][i] = coo[flag][l1];
					csr_column_index_sorted[1 - flag][i++] = csr_column_index_sorted[flag][l1++];
				}
				//if l2 is smaller than csr_row_ptr[pivot + stride * 2], then copy the rest of the elements in the second group
				while (l2 < csr_row_ptr[pivot + stride * 2])
				{
					coo[1 - flag][i] = coo[flag][l2];
					csr_column_index_sorted[1 - flag][i++] = csr_column_index_sorted[flag][l2++];
				}
				//only one of these two while loops will be executed one time
			}
		}
		//the flag here is used to indicate which array is the final sorted array

		int weight = 1;

		int cq = 0, cr = 0;

		// dense bit extract (and ASpT_tile_row_ptr making)
		//iterate row by row in current panel
		for (i = csr_row_ptr[row_panel * BH] + 1; i < csr_row_ptr[(row_panel + 1) * BH]; i++)
		{
			//find the number of non-zero elements in current column
			if (csr_column_index_sorted[flag][i - 1] == csr_column_index_sorted[flag][i])
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
		// int reminder = (csr_column_index_sorted[flag][i-1]&31);
		// check if the number of non-zero elements in the last column is larger than THRESHOLD
		if (weight >= THRESHOLD)
		{
			cr++;
		} // if(cr == BW) { cq++; cr=0; }
		// TODO = occ control
		//total_tile_number is the number of tiles in current row panel, + 1 because all the sparse columns are stored in the last tile
		total_tile_number[row_panel + 1] = CEIL(cr, BW) + 1;
	}

	////gettimeofday(&tt1, NULL);
	// prefix-sum
	for (int i = 1; i <= number_row_panel; i++)
		total_tile_number[i] += total_tile_number[i - 1];
	// ASpT_tile_row_ptr[0] = 0;
	ASpT_tile_row_ptr[BH * total_tile_number[number_row_panel]] = number_nonzero;

	////gettimeofday(&tt2, NULL);

#pragma omp parallel for num_threads(68) schedule(dynamic, 1)
	for (int row_panel = 0; row_panel < number_row / BH; row_panel++)
	{
		int tid = omp_get_thread_num();
		//if currnet row panel is dense
		if (is_row_panel_sparse[row_panel] == 0)
		{
			int i, j;
			int flag = 0;
			int current_tile_id = 0, dense_column_num_in_tile = 0;
			for (int stride = 1; stride <= BH / 2; stride *= 2, flag = 1 - flag)
				;// this for loop is used to find the stride, as well as the flag--which array is the final sorted array
			//row_panel_base_offset is the starting 2D tile row index of the current tile
			int row_panel_base_offset = (total_tile_number[row_panel] * BH);
			//row_panel_num_tile is the number of tiles in current row panel
			int row_panel_num_tile = total_tile_number[row_panel + 1] - total_tile_number[row_panel];
			int weight = 1;

			// ASpT_tile_row_ptr making
			//i iterates all the non-zero elements in current row panel
			for (i = csr_row_ptr[row_panel * BH] + 1; i < csr_row_ptr[(row_panel + 1) * BH]; i++)
			{
				if (csr_column_index_sorted[flag][i - 1] == csr_column_index_sorted[flag][i])
					weight++;
				else
				{
					//reminder is the column index of the last non-zero element in current column % 32
					int reminder = (csr_column_index_sorted[flag][i - 1] & 31);
					if (weight >= THRESHOLD)
					{
						dense_column_num_in_tile++;
						//bit_vector is the bit vector, which is used to mark whether the column index is in the sparse tile or not
						//csr_column_index_sorted[flag][i-1] shifts right by 5 bits, which is the same as dividing by 32
						bit_vector[tid][csr_column_index_sorted[flag][i - 1] >> 5] |= (1 << reminder);
						//j iterates all the non-zero elements in current column
						for (j = i - weight; j <= i - 1; j++)
						{
							//now ASpT_tile_row_ptr documents the number of non-zero elements in each row of each tile
							//now only elements in dense columns are counted
							ASpT_tile_row_ptr[row_panel_base_offset + coo[flag][j] * row_panel_num_tile + current_tile_id + 1]++;
						}
					}
					else
					{
						// bit_vector[tid][csr_column_index_sorted[flag][i-1]>>5] &= (~0 - (1<<reminder));
						bit_vector[tid][csr_column_index_sorted[flag][i - 1] >> 5] &= (0xFFFFFFFF - (1 << reminder));
					}

					//if the number of non-zero elements in current column is larger than BW, then move to the next tile
					//BW is the number of columns in each tile
					if (dense_column_num_in_tile == BW)
					{
						current_tile_id++;
						dense_column_num_in_tile = 0;
					}
					weight = 1;
				}
			}

			// fprintf(stderr, "inter : %d\n", i);

			//process the last elements in current column
			int reminder = (csr_column_index_sorted[flag][i - 1] & 31);
			if (weight >= THRESHOLD)
			{
				dense_column_num_in_tile++;
				bit_vector[tid][csr_column_index_sorted[flag][i - 1] >> 5] |= (1 << reminder);
				for (j = i - weight; j <= i - 1; j++)
				{
					ASpT_tile_row_ptr[row_panel_base_offset + coo[flag][j] * row_panel_num_tile + current_tile_id + 1]++;
				}
			}
			else
			{
				bit_vector[tid][csr_column_index_sorted[flag][i - 1] >> 5] &= (0xFFFFFFFF - (1 << reminder));
			}
			// reordering
			//delta is the number of tiles in current row panel
			int delta = total_tile_number[row_panel + 1] - total_tile_number[row_panel];
			//base0 is the starting 2D tile row index of the current row panel
			int base0 = total_tile_number[row_panel] * BH;
			//i iterate all the rows in current row panel
			for (i = row_panel * BH; i < (row_panel + 1) * BH; i++)
			{
				//row_panel_base_offset is the starting 2D tile index of the current row
				int base = base0 + (i - row_panel * BH) * delta;
				//dense_part_pointer points to the first non-zero element in current row, which is the starting point of first dense tile
				// ASpT_tile_row_ptr[base] = csr_row_ptr[i] = number of elements before current row in current row panel
				int dense_part_pointer = ASpT_tile_row_ptr[base] = csr_row_ptr[i];
				//j iterates all tiles in current row
				for (int j = 1; j < delta; j++)
				{
					//partial prefix sum
					//making ASpT_tile_row_ptr a complete 2D tile row pointer
					ASpT_tile_row_ptr[base + j] += ASpT_tile_row_ptr[base + j - 1];
				}
				//sparse_part_pointer is the starting point of the sparse tile
				int sparse_part_pointer = ASpT_tile_row_ptr[total_tile_number[row_panel] * BH + (total_tile_number[row_panel + 1] - total_tile_number[row_panel]) * (i - row_panel * BH + 1) - 1];

				//csr_row_ptr[i+1] is the start point of the next row
				//csr_row_ptr[i+1] - sparse_part_pointer gives the number of non-zero elements in the sparse tile in current row
				avg0[tid] += csr_row_ptr[i + 1] - sparse_part_pointer;
				//j iterates all the non-zero elements in current row
				for (j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
				{
					//k is the column index of the j-th non-zero element in current row
					int k = csr_column_index[j];
					if ((bit_vector[tid][k >> 5] & (1 << (k & 31))))
					{
						//sort the element in dense tile
						ASpT_csr_column_index[dense_part_pointer] = csr_column_index[j];
						ASpT_csr_values[dense_part_pointer++] = csr_values[j];
					}
					else
					{
						//sort the element in sparse tile
						ASpT_csr_column_index[sparse_part_pointer] = csr_column_index[j];
						ASpT_csr_values[sparse_part_pointer++] = csr_values[j];
					}
				}
			}
		}
		else
		{
			int base0 = total_tile_number[row_panel] * BH;
			memcpy(&ASpT_tile_row_ptr[base0], &csr_row_ptr[row_panel * BH], sizeof(int) * BH);
			avg0[tid] += csr_row_ptr[(row_panel + 1) * BH] - csr_row_ptr[row_panel * BH];
			int bidx = csr_row_ptr[row_panel * BH];
			int bseg = csr_row_ptr[(row_panel + 1) * BH] - bidx;
			//sparse row panel's non-zero elements are stored in CSR format
			memcpy(&ASpT_csr_column_index[bidx], &csr_column_index[bidx], sizeof(int) * bseg);
			memcpy(&ASpT_csr_values[bidx], &csr_values[bidx], sizeof(FTYPE) * bseg);
		}
	}

	for (int i = 0; i < NTHREAD; i++)
		avg_sparse_elem_in_row += avg0[i];
	avg_sparse_elem_in_row /= (double)number_row;
	//avg_sparse_elem_in_row gives the average number of sparse tiles' non-zero elements in each row

	////gettimeofday(&tt3, NULL);

	for (int i = 0; i < number_row; i++)
	{
		//next_row_idx is the index of next row in ASpT_tile_row_ptr
		int next_row_idx = (total_tile_number[i >> LOG_BH]) * BH + (total_tile_number[(i >> LOG_BH) + 1] - total_tile_number[i >> LOG_BH]) * ((i & (BH - 1)) + 1);
		//num_sparse_elem gives the number of sparse tiles' non-zero elements in current row
		int num_sparse_elem = csr_row_ptr[i + 1] - ASpT_tile_row_ptr[next_row_idx - 1];
		//r is the difference between the number of sparse tiles' non-zero elements in current row and the average number of sparse tiles' non-zero elements in each row
		double r = ((double)num_sparse_elem - avg_sparse_elem_in_row);
		//vari_per_row is the variance of the number of sparse tiles' non-zero elements in each row
		vari_per_row += r * r;

		//if the number of sparse tiles' non-zero elements in current row is larger than STHRESHOLD, then split the row
		if (num_sparse_elem >= STHRESHOLD)
		{
			int surplus_times = (num_sparse_elem) / STHRESHOLD;
			for (int j = 0; j < surplus_times; j++)
			{
				//special_row_id is used to store the row id of the split row
				special_row_id[special_p] = i;
				//special_col_id is used to store the column id(of the first non-zero element in the split row) of the split row
				special_col_id[special_p] = j * STHRESHOLD;
				special_p++;
			}
		}
	}
	vari_per_row /= (double)number_row;

	gettimeofday(&endtime, NULL);

	double elapsed0 = ((start_time.tv_sec - starttime0.tv_sec) * 1000000 + start_time.tv_usec - starttime0.tv_usec) / 1000000.0;
	// double elapsed1 = ((tt1.tv_sec-start_time.tv_sec)*1000000 + tt1.tv_usec-start_time.tv_usec)/1000000.0;
	// double elapsed2 = ((tt2.tv_sec-tt1.tv_sec)*1000000 + tt2.tv_usec-tt1.tv_usec)/1000000.0;
	// double elapsed3 = ((tt3.tv_sec-tt2.tv_sec)*1000000 + tt3.tv_usec-tt2.tv_usec)/1000000.0;
	// double elapsed4 = ((endtime.tv_sec-tt3.tv_sec)*1000000 + endtime.tv_usec-tt3.tv_usec)/1000000.0;
	// fprintf(stdout, "(%f %f %f %f %f)", elapsed0*1000, elapsed1*1000, elapsed2*1000, elapsed3*1000, elapsed4*1000);
	//process elapsed time
	p_elapsed = ((endtime.tv_sec - start_time.tv_sec) * 1000000 + endtime.tv_usec - start_time.tv_usec) / 1000000.0;
	fprintf(stdout, "%f,%f,", elapsed0 * 1000, p_elapsed * 1000);

	for (int i = 0; i < NTHREAD; i++)
		free(bit_vector[i]);
	for (int i = 0; i < 2; i++)
	{
		free(csr_column_index_sorted[i]);
		free(coo[i]);
	}
	free(bit_vector);
	free(csr_column_index_sorted);
	free(coo);
}

void ASpT_matrix_multiple()
{
	FILE *fpo = fopen("SpMM_KNL_DP.out", "a");
	FILE *fpo2 = fopen("SpMM_KNL_DP_preprocessing.out", "a");

	double elapsed[3];
	FTYPE *dense_matrix_vector, *result_vector;
	FTYPE *validate_vector;
	dense_matrix_vector = (FTYPE *)_mm_malloc(sizeof(FTYPE) * number_column * num_dense_matrix_column, 64);
	result_vector = (FTYPE *)_mm_malloc(sizeof(FTYPE) * number_row * num_dense_matrix_column, 64);

	__assume_aligned(csr_row_ptr, 64);
	__assume_aligned(ASpT_csr_column_index, 64);
	__assume_aligned(ASpT_csr_values, 64);
	//__assume_aligned(total_tile_number, 64);
	//__assume_aligned(ASpT_tile_row_ptr, 64);
	//__assume_aligned(mcsr_list, 64);
	__assume_aligned(dense_matrix_vector, 64);
	__assume_aligned(result_vector, 64);

	//num_dense_matrix_column is the column number of the dense matrix
	for (num_dense_matrix_column = 8; num_dense_matrix_column <= 128; num_dense_matrix_column *= 4)
	{

		struct timeval starttime, endtime;

		//initialize dense_matrix_vector and result_vector
		//result_vector is the output matrix(in the form of 1D array)
		memset(result_vector, 0, sizeof(FTYPE) * number_row * num_dense_matrix_column);
#pragma vector aligned
#pragma omp parallel for num_threads(68)
		for (int i = 0; i < number_column * num_dense_matrix_column; i++)
		{
			//dense_matrix_vector is the dense matrix(in the form of 1D array)
			dense_matrix_vector[i] = (FTYPE)(rand() % 1048576) / 1048576;
#ifdef SIM_VALUE
			dense_matrix_vector[i] = 1;
#endif
		}

		// double tot_time;
// cout << "v" << ne/nc << endl;
#define ITER (128 * 1 / 128)
		if (vari_per_row < 5000 * 1 / 1 * 1)
		{

			gettimeofday(&starttime, NULL);
			////begin
			//calculate the output matrix ITER times and add it to result_vector
			for (int loop = 0; loop < ITER; loop++)
			{
#pragma ivdep //ivdep is a hint to the compiler that the loop is safe for instruction-level parallelism
#pragma vector aligned //vector aligned is a hint to the compiler that the loop is safe for vectorization
#pragma temporal(vin) //temporal is a hint to the compiler that the loop is safe for temporal locality
#pragma omp parallel for num_threads(136) schedule(dynamic, 1)
				for (int row_panel = 0; row_panel < number_row / BH; row_panel++)
				{
					// dense
					int stride;
					//stride iterates all the dense tiles in current row panel
					for (stride = 0; stride < total_tile_number[row_panel + 1] - total_tile_number[row_panel] - 1; stride++)
					{
						
						//i iterates all the rows in current dense tile, which is also the row id in sparse matrix
						for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++)
						{
							//tile_row_first_elem is the ASpT_tile_row_ptr	index of the first non-zero element in current dense tile in current row
							int tile_row_first_elem = total_tile_number[row_panel] * BH + (i & (BH - 1)) * (total_tile_number[row_panel + 1] - total_tile_number[row_panel]) + stride;
							//loc1 is the column index of the first non-zero element in current dense tile in current row
							//loc2 is the column index of the last non-zero element in current dense tile in current row + 1
							int loc1 = ASpT_tile_row_ptr[tile_row_first_elem], loc2 = ASpT_tile_row_ptr[tile_row_first_elem + 1];

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
								
								for (int k = 0; k < num_dense_matrix_column; k++)
								{
									result_vector[i * num_dense_matrix_column + k] = result_vector[i * num_dense_matrix_column + k] + ASpT_csr_values[j] * dense_matrix_vector[ASpT_csr_column_index[j] * num_dense_matrix_column + k] + ASpT_csr_values[j + 1] * dense_matrix_vector[ASpT_csr_column_index[j + 1] * num_dense_matrix_column + k] + ASpT_csr_values[j + 2] * dense_matrix_vector[ASpT_csr_column_index[j + 2] * num_dense_matrix_column + k] + ASpT_csr_values[j + 3] * dense_matrix_vector[ASpT_csr_column_index[j + 3] * num_dense_matrix_column + k] + ASpT_csr_values[j + 4] * dense_matrix_vector[ASpT_csr_column_index[j + 4] * num_dense_matrix_column + k] + ASpT_csr_values[j + 5] * dense_matrix_vector[ASpT_csr_column_index[j + 5] * num_dense_matrix_column + k] + ASpT_csr_values[j + 6] * dense_matrix_vector[ASpT_csr_column_index[j + 6] * num_dense_matrix_column + k] + ASpT_csr_values[j + 7] * dense_matrix_vector[ASpT_csr_column_index[j + 7] * num_dense_matrix_column + k];
								}
							}
							for (; j < loc2; j++)
							{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vout : _MM_HINT_T1
#pragma temporal(vout)
								for (int k = 0; k < num_dense_matrix_column; k++)
								{
									result_vector[i * num_dense_matrix_column + k] += ASpT_csr_values[j] * dense_matrix_vector[ASpT_csr_column_index[j] * num_dense_matrix_column + k];
								}
							}
						}
					}
					// sparse
					for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++)
					{
						//the sparse part is stored in CSR format
						//now stride is the index of the sparse tile in current row panel
						//tile_row_first_elem here is the ASpT_tile_row_ptr index of the first non-zero element in current sparse tile in current row
						int tile_row_first_elem = total_tile_number[row_panel] * BH + (i & (BH - 1)) * (total_tile_number[row_panel + 1] - total_tile_number[row_panel]) + stride;
						int loc1 = ASpT_tile_row_ptr[tile_row_first_elem], loc2 = ASpT_tile_row_ptr[tile_row_first_elem + 1];

						// printf("(%d %d %d %d %d)\n", i, csr_row_ptr[i], loc1, csr_row_ptr[i+1], loc2);
						// printf("%d %d %d %d %d %d %d\n", i, tile_row_first_elem, stride, csr_row_ptr[i], loc1, csr_row_ptr[i+1], loc2);

						int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
						int j;
						for (j = loc1; j < interm; j += 8)
						{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vin : _MM_HINT_T1
#pragma temporal(vin)
							for (int k = 0; k < num_dense_matrix_column; k++)
							{
								result_vector[i * num_dense_matrix_column + k] = result_vector[i * num_dense_matrix_column + k] + ASpT_csr_values[j] * dense_matrix_vector[ASpT_csr_column_index[j] * num_dense_matrix_column + k] + ASpT_csr_values[j + 1] * dense_matrix_vector[ASpT_csr_column_index[j + 1] * num_dense_matrix_column + k] + ASpT_csr_values[j + 2] * dense_matrix_vector[ASpT_csr_column_index[j + 2] * num_dense_matrix_column + k] + ASpT_csr_values[j + 3] * dense_matrix_vector[ASpT_csr_column_index[j + 3] * num_dense_matrix_column + k] + ASpT_csr_values[j + 4] * dense_matrix_vector[ASpT_csr_column_index[j + 4] * num_dense_matrix_column + k] + ASpT_csr_values[j + 5] * dense_matrix_vector[ASpT_csr_column_index[j + 5] * num_dense_matrix_column + k] + ASpT_csr_values[j + 6] * dense_matrix_vector[ASpT_csr_column_index[j + 6] * num_dense_matrix_column + k] + ASpT_csr_values[j + 7] * dense_matrix_vector[ASpT_csr_column_index[j + 7] * num_dense_matrix_column + k];
							}
						}
						for (; j < loc2; j++)
						{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vout : _MM_HINT_T1
#pragma temporal(vout)
							for (int k = 0; k < num_dense_matrix_column; k++)
							{
								result_vector[i * num_dense_matrix_column + k] += ASpT_csr_values[j] * dense_matrix_vector[ASpT_csr_column_index[j] * num_dense_matrix_column + k];
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
				for (int row_panel = 0; row_panel < number_row / BH; row_panel++)
				{
					// dense
					int stride;
					for (stride = 0; stride < total_tile_number[row_panel + 1] - total_tile_number[row_panel] - 1; stride++)
					{

						for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++)
						{
							int tile_row_first_elem = total_tile_number[row_panel] * BH + (i & (BH - 1)) * (total_tile_number[row_panel + 1] - total_tile_number[row_panel]) + stride;
							int loc1 = ASpT_tile_row_ptr[tile_row_first_elem], loc2 = ASpT_tile_row_ptr[tile_row_first_elem + 1];

							int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
							int j;
							for (j = loc1; j < interm; j += 8)
							{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vin : _MM_HINT_T1
#pragma temporal(vin)
								for (int k = 0; k < num_dense_matrix_column; k++)
								{
									result_vector[i * num_dense_matrix_column + k] = result_vector[i * num_dense_matrix_column + k] + ASpT_csr_values[j] * dense_matrix_vector[ASpT_csr_column_index[j] * num_dense_matrix_column + k] + ASpT_csr_values[j + 1] * dense_matrix_vector[ASpT_csr_column_index[j + 1] * num_dense_matrix_column + k] + ASpT_csr_values[j + 2] * dense_matrix_vector[ASpT_csr_column_index[j + 2] * num_dense_matrix_column + k] + ASpT_csr_values[j + 3] * dense_matrix_vector[ASpT_csr_column_index[j + 3] * num_dense_matrix_column + k] + ASpT_csr_values[j + 4] * dense_matrix_vector[ASpT_csr_column_index[j + 4] * num_dense_matrix_column + k] + ASpT_csr_values[j + 5] * dense_matrix_vector[ASpT_csr_column_index[j + 5] * num_dense_matrix_column + k] + ASpT_csr_values[j + 6] * dense_matrix_vector[ASpT_csr_column_index[j + 6] * num_dense_matrix_column + k] + ASpT_csr_values[j + 7] * dense_matrix_vector[ASpT_csr_column_index[j + 7] * num_dense_matrix_column + k];
								}
							}
							for (; j < loc2; j++)
							{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vout : _MM_HINT_T1
#pragma temporal(vout)
								for (int k = 0; k < num_dense_matrix_column; k++)
								{
									result_vector[i * num_dense_matrix_column + k] += ASpT_csr_values[j] * dense_matrix_vector[ASpT_csr_column_index[j] * num_dense_matrix_column + k];
								}
							}
						}
					}
					// sparse
					for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++)
					{

						int tile_row_first_elem = total_tile_number[row_panel] * BH + (i & (BH - 1)) * (total_tile_number[row_panel + 1] - total_tile_number[row_panel]) + stride;
						int loc1 = ASpT_tile_row_ptr[tile_row_first_elem], loc2 = ASpT_tile_row_ptr[tile_row_first_elem + 1];

						loc1 += ((loc2 - loc1) / STHRESHOLD) * STHRESHOLD;

						int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
						int j;
						for (j = loc1; j < interm; j += 8)
						{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vin : _MM_HINT_T1
#pragma temporal(vin)
							for (int k = 0; k < num_dense_matrix_column; k++)
							{
								result_vector[i * num_dense_matrix_column + k] = result_vector[i * num_dense_matrix_column + k] + ASpT_csr_values[j] * dense_matrix_vector[ASpT_csr_column_index[j] * num_dense_matrix_column + k] + ASpT_csr_values[j + 1] * dense_matrix_vector[ASpT_csr_column_index[j + 1] * num_dense_matrix_column + k] + ASpT_csr_values[j + 2] * dense_matrix_vector[ASpT_csr_column_index[j + 2] * num_dense_matrix_column + k] + ASpT_csr_values[j + 3] * dense_matrix_vector[ASpT_csr_column_index[j + 3] * num_dense_matrix_column + k] + ASpT_csr_values[j + 4] * dense_matrix_vector[ASpT_csr_column_index[j + 4] * num_dense_matrix_column + k] + ASpT_csr_values[j + 5] * dense_matrix_vector[ASpT_csr_column_index[j + 5] * num_dense_matrix_column + k] + ASpT_csr_values[j + 6] * dense_matrix_vector[ASpT_csr_column_index[j + 6] * num_dense_matrix_column + k] + ASpT_csr_values[j + 7] * dense_matrix_vector[ASpT_csr_column_index[j + 7] * num_dense_matrix_column + k];
							}
						}
						for (; j < loc2; j++)
						{
#pragma ivdep
#pragma vector nontemporal(csr_ev)
#pragma prefetch vout : _MM_HINT_T1
#pragma temporal(vout)
							for (int k = 0; k < num_dense_matrix_column; k++)
							{
								result_vector[i * num_dense_matrix_column + k] += ASpT_csr_values[j] * dense_matrix_vector[ASpT_csr_column_index[j] * num_dense_matrix_column + k];
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
					int i = special_row_id[row_panel];

					int tile_row_first_elem = total_tile_number[i >> LOG_BH] * BH + ((i & (BH - 1)) + 1) * (total_tile_number[(i >> LOG_BH) + 1] - total_tile_number[i >> LOG_BH]);

					int loc1 = ASpT_tile_row_ptr[tile_row_first_elem - 1] + special_col_id[row_panel];
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
						for (int k = 0; k < num_dense_matrix_column; k++)
						{
							temp_r[k] = temp_r[k] + ASpT_csr_values[j] * dense_matrix_vector[ASpT_csr_column_index[j] * num_dense_matrix_column + k] + ASpT_csr_values[j + 1] * dense_matrix_vector[ASpT_csr_column_index[j + 1] * num_dense_matrix_column + k] + ASpT_csr_values[j + 2] * dense_matrix_vector[ASpT_csr_column_index[j + 2] * num_dense_matrix_column + k] + ASpT_csr_values[j + 3] * dense_matrix_vector[ASpT_csr_column_index[j + 3] * num_dense_matrix_column + k] + ASpT_csr_values[j + 4] * dense_matrix_vector[ASpT_csr_column_index[j + 4] * num_dense_matrix_column + k] + ASpT_csr_values[j + 5] * dense_matrix_vector[ASpT_csr_column_index[j + 5] * num_dense_matrix_column + k] + ASpT_csr_values[j + 6] * dense_matrix_vector[ASpT_csr_column_index[j + 6] * num_dense_matrix_column + k] + ASpT_csr_values[j + 7] * dense_matrix_vector[ASpT_csr_column_index[j + 7] * num_dense_matrix_column + k];
						}
					}
#pragma ivdep
					for (int k = 0; k < num_dense_matrix_column; k++)
					{
#pragma omp atomic
						result_vector[i * num_dense_matrix_column + k] += temp_r[k];
					}
				}

			} // end loop
			gettimeofday(&endtime, NULL);
		}

		// double elapsed = ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
		if (num_dense_matrix_column == 8)
			elapsed[0] = ((endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec) / 1000000.0;
		else if (num_dense_matrix_column == 32)
			elapsed[1] = ((endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec) / 1000000.0;
		else
			elapsed[2] = ((endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec) / 1000000.0;
	}
	num_dense_matrix_column = 128;
	validate_vector = (FTYPE *)malloc(sizeof(FTYPE) * number_row * num_dense_matrix_column);
	memset(validate_vector, 0, sizeof(FTYPE) * number_row * num_dense_matrix_column);

#define VALIDATE
#if defined VALIDATE
	// validate
	for (int i = 0; i < number_row * num_dense_matrix_column; i++)
	{
		validate_vector[i] = 0.0f;
	}
	for (num_dense_matrix_column = 128; num_dense_matrix_column <= 128; num_dense_matrix_column *= 4)
	{
		for (int i = 0; i < validate_num_element; i++)
		{
			for (int j = 0; j < num_dense_matrix_column; j++)
			{
				//validate_vector stored the correct output matrix
				validate_vector[validate_temp_matrix_vector[i].row * num_dense_matrix_column + j] += dense_matrix_vector[num_dense_matrix_column * validate_temp_matrix_vector[i].col + j] * validate_temp_matrix_vector[i].val;
			}
		}
	}
	num_dense_matrix_column = 128;
	int num_diff = 0;
	//the loop iterates all the elements in the output matrix
	for (int i = 0; i < number_row * num_dense_matrix_column; i++)
	{
		//validate_vector is multiplied by ITER, because the output matrix is calculated ITER times
		FTYPE p1 = validate_vector[i] * ITER;
		FTYPE p2 = result_vector[i];

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
			// if(num_diff < 20*1*1) fprintf(stdout, "%d %f %f\n", i, result_vector[i], validate_vector[i]*ITER);
			// if(result_vector[i] < validate_vector[i]) fprintf(stdout, "%d %f %f\n", i, result_vector[i], validate_vector[i]);

			num_diff++;
		}
	}
	//      fprintf(stdout, "num_diff : %d\n", num_diff);
	//print the percentage of different elements
	fprintf(stdout, "%f,", (double)num_diff / (number_row * num_dense_matrix_column) * 100);
//      fprintf(stdout, "ne : %d\n", validate_num_element);
#endif
	// fprintf(stdout, "%f,%f\n", (double)elapsed*1000/ITER, (double)ne*2*num_dense_matrix_column*ITER/elapsed/1000000000);
	fprintf(stdout, "%f,", (double)number_nonzero * 2 * 8 * ITER / elapsed[0] / 1000000000);
	fprintf(stdout, "%f,%f,", (double)number_nonzero * 2 * 32 * ITER / elapsed[1] / 1000000000, (double)number_nonzero * 2 * 128 * ITER / elapsed[2] / 1000000000);
	//the ratio of the time of preprocessing to the time of calculating the output matrix
	fprintf(stdout, "%f\n", p_elapsed / (elapsed[2] / ITER));

	//ASpT GFLOPS(K=32,K=128)
	fprintf(fpo, "%f,%f,", (double)number_nonzero * 2 * 32 * ITER / elapsed[1] / 1000000000, (double)number_nonzero * 2 * 128 * ITER / elapsed[2] / 1000000000);
	//num_diff (K=32+K=128)
	fprintf(fpo, "%f,", (double)num_diff / (number_row * num_dense_matrix_column) * 100);
	fprintf(fpo2, "%f", p_elapsed / (elapsed[2] / ITER));
	fclose(fpo);
	fclose(fpo2);
}

int main(int argc, char **argv)
{
	load_matrix_into_CSR(argc, argv);
	ASpT_preprocess();
	// gen_structure();
	ASpT_matrix_multiple();
}
