#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "linear_regression.h"

static void fit(linear_regression* m, array* x, array* y){
	PyObject* PY_x = PyObject_from_double_array(x);
	PyObject* PY_y = PyObject_from_double_array(x);
	
	PyObject* args = build_arguments(2, PY_x, PY_y);
	
	PyObject* fitted_estimator = call_method(m->self, "fit", args, NULL);
	
	Py_DECREF(args);
	
	if (fitted_estimator == NULL) {
		Py_DECREF(fitted_estimator);
		error("An error ocurred when calling fit with x[] and/or y[]");
	}
	Py_DECREF(PY_x); Py_DECREF(PY_y);
		
	/*
	PyObject* coef_ = get_attribute(m->self, "coef_");
	PyObject* rank_ = get_attribute(m->self, "rank_");
	PyObject* singular_ = get_attribute(m->self, "singular_");
	PyObject* intercept_ = get_attribute(m->self, "intercept_");
	PyObject* n_features_in_ = get_attribute(m->self, "n_features_in_");

	PyObject* feature_names_in_;
	if (PyObject_HasAttrString(m->self, "feature_names_in_")){
		feature_names_in_ = get_attribute(m->self, "feature_names_in_");
	}
	*/
	
	// parse attributes and fill attributes struct
	
	Py_DECREF(fitted_estimator);
}

static void get_params(linear_regression* m){
	PyObject* params = call_method(m->self, "get_params", NULL, NULL);
	if (params == NULL) {
		Py_DECREF(params);
		error("An error ocurred with scikitlearn!");
	}
	reprint(params);
	Py_DECREF(params);
}

static array* predict(linear_regression* m, array* x){
	PyObject* PY_x = PyObject_from_double_array(x);

	PyObject* args = build_arguments(1, PY_x);

	PyObject* prediction = call_method(m->self, "predict", args, NULL);
	if (prediction == NULL) {
		Py_DECREF(prediction);
		error("An error ocurred when calling predict with x[]");
	}
	
	Py_DECREF(args);
	
	// parse and return prediction

	if (!PyArray_Check(prediction)) {error("Not an array");}
	PyArrayObject* numpy_array = PyObject_to_PyArrayObject(prediction);
	reprint(numpy_array);		

	Py_DECREF(prediction);
	
	return NULL;
}

// NOTE: doesn't support sample_weight
static double score(linear_regression* m, array* x, array* y){
	PyObject* PY_x = PyObject_from_double_array(x);
	PyObject* PY_y = PyObject_from_double_array(y);
	PyObject* args = build_arguments(2, PY_x, PY_y);
	PyObject* score = call_method(m->self, "score", args, NULL);
	if (score == NULL) {
		Py_DECREF(score);
		error("An error ocurred when calling score with x[] and/or y[]");
	}
	Py_DECREF(args);
	double score_c = double_from_PyObject(score);
	Py_DECREF(score);
	return score_c;
}

static void set_params(linear_regression* m){
	PyObject *params = PyDict_New();
	PyDict_SetItem(params, PyUnicode_FromString("copy_X"), PyObject_from_boolean_int(m->parameters.copy_X));
    PyDict_SetItem(params, PyUnicode_FromString("fit_intercept"), PyObject_from_boolean_int(m->parameters.fit_intercept));
    PyDict_SetItem(params, PyUnicode_FromString("n_jobs"), PyObject_from_int(m->parameters.n_jobs));
    PyDict_SetItem(params, PyUnicode_FromString("positive"), PyObject_from_boolean_int(m->parameters.positive)); 
    PyObject *res = call_method(m->self, "set_params", NULL, params);
	if (res == NULL) {
		Py_DECREF(res);
		error("An error ocurred while calling set_params!");
	}
	Py_DECREF(res);
    Py_DECREF(params);
}

static void purge(linear_regression* m){
	Py_DECREF(m->self);
	free(m);
	finalize_python();
}

static void get_parameter_defaults(linear_regression* m){	
    m->parameters.fit_intercept = 1;
    m->parameters.copy_X		= 1;
    m->parameters.n_jobs		= 1;
    m->parameters.positive      = 0;
}

linear_regression* get_linear_regression(){
	initialize_python();
	linear_regression* m = malloc(sizeof(linear_regression));
	m->self = get_class_instance("sklearn.linear_model","LinearRegression", NULL);
  	get_parameter_defaults(m);
  	m->fit 				 = &fit;
  	m->get_params 	     = &get_params;
  	m->predict 		     = &predict;
  	m->score 		     = &score;
  	m->set_params 		 = &set_params;
  	m->purge             = &purge;
  	return m;
}



