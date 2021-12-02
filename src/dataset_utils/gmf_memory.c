/*
 * gmf_memory.c
 *
 *  Created on: Aug 7, 2021
 *      Author: saul
 */
#include <stdio.h>
#include <stdlib.h>

#include "gmf_global.h"

void gmf_alloc_solution(gmf_solution *sol) {
	sol->f = malloc(sizeof(double) * gmf_mop.nobjs);
    sol->fp = malloc(sizeof(double) * gmf_mop.nobjs);
	sol->xr = malloc(sizeof(double) * gmf_mop.nreal);
}

void gmf_alloc_test(gmf_test *test){
	test->xr = malloc(sizeof(double) * gmf_mop.nreal);
}

void gmf_free_solution(gmf_solution *sol) {
	free(sol->f);
    free(sol->fp);
	free(sol->xr);
}

void gmf_free_test(gmf_test *test){
	free(test->xr);
}

void gmf_alloc_population(gmf_population *pop, int size) {
	int i;
	pop->size = size;
	pop->solutions = (gmf_solution*) malloc(sizeof(gmf_solution) * size);
	for (i = 0; i < size; ++i)
	{
		gmf_alloc_solution(&pop->solutions[i]);
	}
}

void gmf_alloc_population_test(gmf_population_test *pop, int size){
	int i;
	pop->size = size;
	pop->tests = (gmf_test*) malloc(sizeof(gmf_test) * size);
	for (i = 0; i < size; ++i)
	{
		gmf_alloc_test(&pop->tests[i]);
	}
}

void gmf_free_population(gmf_population *pop) {
	int i;
	for (i = 0; i < pop->size; ++i)
	{
		gmf_free_solution(&pop->solutions[i]);
	}

	free(pop->solutions);
}

void gmf_free_population_test(gmf_population_test *pop){
	int i;
	for (i = 0; i < pop->size; ++i)
	{
		gmf_free_test(&pop->tests[i]);
	}

	free(pop->tests);
}

void gmf_rand_pop(gmf_population *P) {
	int i, j, k;
	for (i = 0; i < P->size; ++i)
	{
		for (j = 0; j < (int)gmf_mop.nreal; ++j) // variables
		{
			P->solutions[i].xr[j] = (double) rand()/RAND_MAX;
		}
		for (k = 0; k < (int)gmf_mop.nobjs; ++k) // objetivos
		{
			P->solutions[i].f[k] = (double) rand()/RAND_MAX;
		}
	}
}

void gmf_print_pop(gmf_population *P) {
	int i, j, k;
	for (i = 0; i < P->size; ++i)
	{
		printf("x: ");
		for (j = 0; j < (int)gmf_mop.nreal; ++j) // variables
		{
			printf("%lf ", P->solutions[i].xr[j]);
		}
		printf("\n");
		printf("f: ");
		for (k = 0; k < (int)gmf_mop.nobjs; ++k) // objetivos
		{
			printf("%lf ", P->solutions[i].f[k]);
		}
		printf("\n\n");
	}
}

void gmf_print_pop_test(gmf_population_test *P) {
	int i, j;
	for (i = 0; i < P->size; ++i)
	{
		printf("x: ");
		for (j = 0; j < (int)gmf_mop.nreal; ++j) // variables
		{
			printf("%lf ", P->tests[i].xr[j]);
		}
		printf("\n\n");
	}
}
