#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "skl_svr.h"

static void fit(skl_svr* m, array* x, array* y){

	PyObject* PY_x = PyObject_from_double_array(x);
	PyObject* PY_y = PyObject_from_double_list(y->x[0], y->c);

	PyObject* args = build_arguments(2, PY_x, PY_y);
	
	PyObject* fitted_estimator = call_method(m->self, "fit", args, NULL);

	if (fitted_estimator == NULL) {
		error("An error ocurred when calling fit with x[] and/or y[]");
	}
	
	PyArrayObject* class_weight_ = PyObject_to_PyArrayObject(get_attribute(m->self, "class_weight_"));
	PyArrayObject* coef_ = PyObject_to_PyArrayObject(get_attribute(m->self, "coef_"));
	PyArrayObject* dual_coef_ = PyObject_to_PyArrayObject(get_attribute(m->self, "dual_coef_"));
	PyObject* fit_status_ = get_attribute(m->self, "fit_status_");
	PyArrayObject* intercept_ = PyObject_to_PyArrayObject(get_attribute(m->self, "intercept_"));
	PyObject* n_features_in_ = get_attribute(m->self, "n_features_in_");
	PyArrayObject* feature_names_in_ = PyObject_to_PyArrayObject(get_attribute(m->self, "feature_names_in_"));
	PyArrayObject* n_support_ = PyObject_to_PyArrayObject(get_attribute(m->self, "n_support_"));
	PyArrayObject* shape_fit_ = PyObject_to_PyArrayObject(get_attribute(m->self, "shape_fit_"));
	PyArrayObject* support_ = PyObject_to_PyArrayObject(get_attribute(m->self, "support_"));
	PyArrayObject* support_vectors_ = PyObject_to_PyArrayObject(get_attribute(m->self, "support_vectors_"));
	
	/*
		array* class_weight_;
		array* coef_;
		array* dual_coef_;
		int    fit_status_;
		array* intercept_;
		int    n_features_in_;
		array* feature_names_in_;
		array* n_support_;
		array* shape_fit_;
		array* support_;
		array* support_vectors_;
	*/
	

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
	m->self = get_class_instance("sklearn.svm","SVR", NULL);	
  	get_parameter_defaults(m);
  	m->fit 				 = &fit;
  	m->get_params 	     = &get_params;
  	m->predict 		     = &predict;
  	m->score 		     = &score;
  	m->set_params 		 = &set_params;
  	m->purge             = &purge;
  	return m;
}



