/*
 * gmf_global.h
 *
 *  Created on: Aug 7, 2020
 *      Author: saul
 */

#ifndef GMF_GLOBAL_H_
#define GMF_GLOBAL_H_


#define gmf_true 1
#define gmf_false 0

#include <stddef.h>

typedef struct {
	double *xr;		// Decision parameters
} gmf_test;

typedef struct {
	double *xr;		// Decision parameters
	double *f;		// Objective values
	double *fp;		// Objective values
} gmf_solution;

typedef struct {
	gmf_solution *solutions;	// Array of solutions
	int size;					// Population size
} gmf_population;

typedef struct {
	gmf_test *tests;			// Array of tests
	int size;					// Population size
} gmf_population_test;

struct {
	size_t nreal;	// Number of decision parameters
	size_t nobjs;	// Number of objectives
} gmf_mop;

struct {
	char name[50];
	int type;
	int psize;
} gmf_algorithm;


/* Functions headears */
void gmf_alloc_solution(gmf_solution *sol);
void gmf_alloc_test(gmf_test *test);
void gmf_free_solution(gmf_solution *sol);
void gmf_free_test(gmf_test *test);
void gmf_alloc_population(gmf_population *pop, int size);
void gmf_alloc_population_test(gmf_population_test *pop, int size);
void gmf_free_population(gmf_population *pop);
void gmf_free_population_test(gmf_population_test *pop);

void gmf_rand_pop(gmf_population *P);
void gmf_print_pop(gmf_population *P);
void gmf_print_pop_test(gmf_population_test *Q);

#endif /* GMF_GLOBAL_H_ */
