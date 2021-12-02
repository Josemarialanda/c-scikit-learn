#ifndef DATASET_READ_H_
#define DATASET_READ_H_

#include <stdio.h>
#include "gmf_global.h"
#define BUFFER_SIZE 1024

int count_lines (FILE *fp);

// read the first two lines of dataset which contain info about:
// * number of objectives
// * number of decision variables
void fill_header(char* buffer);

// fill population P with training set data from file fp
void fill_training_set(FILE* fp, gmf_population *P);

// fill population Q with testing set data from file fp
void fill_testing_set(FILE* fp, gmf_population_test *Q);

FILE* read_file(char* src);

// read training set and if file exists fill population P
void read_training_set(char* src, gmf_population* P);

// read testing set and if file exists fill population Q
void read_testing_set(char* src, gmf_population_test* Q);

#endif /* DATASET_READ_H_ */
