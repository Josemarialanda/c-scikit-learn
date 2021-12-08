#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "skl_svr.h"

static void fit(skl_svr* m, array* x, array* y){

}

static void get_params(skl_svr* m){

}

static array* predict(skl_svr* m, array* x){
	return NULL;
}

static double score(skl_svr* m, array* x, array* y){
	return 1;
}

static void set_params(skl_svr* m){
	
}

static void purge(skl_svr* m){
	Py_DECREF(m->self);
	free(m);
	finalize_python();
}

static void get_parameter_defaults(skl_svr* m){	
    m->parameters.kernel     = "rbf";
    m->parameters.degree     = 3;
    m->parameters.gamma      = "scale"; // for now doesn't take float as an option
    m->parameters.coef0      = 0;
    m->parameters.tol        = 0.001;
    m->parameters.C          = 1;
    m->parameters.epsilon    = 0.1;
    m->parameters.shrinking  = 1;
    m->parameters.cache_size = 200;
    m->parameters.verbose    = 0;
    m->parameters.max_iter   = -1;
}

skl_svr* skl_get_svr(){
	initialize_python();
	skl_svr* m = malloc(sizeof(skl_svr));
	// m->self = get_class_instance("sklearn.linear_model","LinearRegression", NULL);
  	get_parameter_defaults(m);
  	m->fit 				 = &fit;
  	m->get_params 	     = &get_params;
  	m->predict 		     = &predict;
  	m->score 		     = &score;
  	m->set_params 		 = &set_params;
  	m->purge             = &purge;
  	return m;
}



