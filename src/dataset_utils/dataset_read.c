#include <stdlib.h>
#include <string.h>
#include <sysexits.h>
#include <ctype.h>

#include "dataset_read.h"

int count_lines (FILE *fp) {
    int  nlines  = 0;
    char line[BUFFER_SIZE];
    while(fgets(line, BUFFER_SIZE, fp) != NULL) {
        nlines++;
    }
    rewind(fp);
    return nlines;
}

void fill_header(char* buffer){
    char* p = buffer;
    while (*p) {
        if (isdigit(*p) && *(p+2) == 'o') {
            int n = (int)strtol(p, &p, 10);
            gmf_mop.nobjs = n;
        } else if (isdigit(*p) && *(p+2) == 'd') {
            int n = (int)strtol(p, &p, 10);
            gmf_mop.nreal = n; 
        }
        else {p++;}
    }
}

void fill_training_set(FILE* fp, gmf_population *P) {
    int pop_size = count_lines(fp) - 2;
    int line_num = 0;
    char buffer[BUFFER_SIZE];             
    while((fgets (buffer, BUFFER_SIZE, fp))!= NULL) {
        if (line_num == 0){
            fill_header(buffer);                     
            gmf_alloc_population(P, pop_size);
            line_num++;
        } else if (line_num == 1) {line_num++; continue;}
        else {            
            char* token = strtok (buffer," ");
            int k = 0;
            while (token != NULL) {
                double d;
                d = strtod (token, &token);
                if (k<(int)gmf_mop.nobjs) {
                    int i = line_num-2;
                    int j = k;
                    P->solutions[i].f[j] = d;
                    P->solutions[i].fp[j] = d;
                    k++;
                } else { 
                    int i = line_num-2;
                    int j = k-(int)gmf_mop.nobjs;
                    P->solutions[i].xr[j] = d;
                    k++;
                }
                token = strtok (NULL, " ");
            }
            line_num++;
        }
    }   
}

void fill_testing_set(FILE* fp, gmf_population_test *Q) {
    int pop_size = count_lines(fp) - 2;
    int line_num = 0;
    char buffer[BUFFER_SIZE];             
    while((fgets (buffer, BUFFER_SIZE, fp))!= NULL) {
        if (line_num == 0){
            fill_header(buffer);
            line_num++;
            gmf_alloc_population_test(Q, pop_size);
        } else if (line_num == 1) {line_num++; continue;}
        else {            
            char* token = strtok (buffer," ");
            int k = 0;
            while (token != NULL) {
                double d;
                d = strtod (token, &token);
                int i = line_num-2;
                Q->tests[i].xr[k] = d;
                k++;
                token = strtok (NULL, " ");
            }
            line_num++;
        }
    }   
}

FILE* read_file(char* src){
    char src_path[100] = "./";
    strcat(src_path, src);
    FILE* fp = fopen(src_path, "r");
    return fp;
}

void read_training_set(char* src, gmf_population* P){
	FILE* fp = read_file(src);
	if (fp){
	    fill_training_set(fp, P);
	}
	else {
		exit(66);
	}
	fclose(fp);
}

void read_testing_set(char* src, gmf_population_test* Q){
    FILE* fp = read_file(src);
	if (fp){
	    fill_testing_set(fp, Q);
	}
	else {
		exit(66);
	}
	fclose(fp);
}
