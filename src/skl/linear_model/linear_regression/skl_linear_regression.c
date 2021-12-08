#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "skl_linear_regression.h"

static void fit(skl_linear_regression* m, array* x, array* y){

	// converts array arguments to Python objects
	PyObject* PY_x = PyObject_from_double_array(x);
	PyObject* PY_y = PyObject_from_double_array(x);
	
	// build arguments for fit with PY_x and PY_y
	PyObject* args = build_arguments(2, PY_x, PY_y);
	
	// call fit with PyObject* args and NULL kwargs
	// note: no need to call Py_DECREF(args) or Py_DECREF(kwargs)
	// call_method() takes care of those references
	
	PyObject* fitted_estimator = call_method(m->self, "fit", args, NULL);
	
	if (fitted_estimator == NULL) {
		error("An error ocurred when calling fit with x[] and/or y[]");
	}
	
	// note: there is no need to decrease reference count of PY_x,PY_y
	// since build_arguments steals the reference from PY_x,PY_y
	// thus and when we call call_method(), we do Py_DECREF(args),
	// which also decreases the reference count for PY_x,PY_y
	
	// fetch attributes from class_instance
	// note: some attributes could return NULL if not defined for certain dataset
	PyArrayObject* coef_ = PyObject_to_PyArrayObject(get_attribute(m->self, "coef_"));
	PyObject* rank_ = get_attribute(m->self, "rank_");
	PyArrayObject* singular_ = PyObject_to_PyArrayObject(get_attribute(m->self, "singular_"));
	PyArrayObject* intercept_ = PyObject_to_PyArrayObject(get_attribute(m->self, "intercept_"));
	PyObject* n_features_in_ = get_attribute(m->self, "n_features_in_");
	PyArrayObject* feature_names_in_ = PyObject_to_PyArrayObject(get_attribute(m->self, "feature_names_in_"));
	
	// fill attributes struct
	
	// always defined
	m->attributes.coef_ = PyArrayObject_to_array(coef_);
	m->attributes.intercept_ = PyArrayObject_to_array(intercept_);
	m->attributes.n_features_in_ = int_from_PyObject(n_features_in_);
	
	// may be NULL
	m->attributes.rank_ = -1;
	m->attributes.singular_ = NULL;
	m->attributes.feature_names_in_ = NULL;
	
	// only available when X is dense
	if (rank_ != NULL) { m->attributes.rank_ = int_from_PyObject(rank_); }
	// only available when X is dense
	if (singular_ != NULL) { m->attributes.singular_ = PyArrayObject_to_array(singular_); }
	// defined only when X has feature names that are all strings
	if (feature_names_in_ != NULL) { m->attributes.feature_names_in_ = PyArrayObject_to_array(feature_names_in_); }
	
	Py_DECREF(coef_);
	Py_DECREF(intercept_);
	Py_DECREF(n_features_in_);
	Py_XDECREF(rank_);
	Py_XDECREF(singular_);
	Py_XDECREF(feature_names_in_);
	
	Py_DECREF(fitted_estimator);
}

static void get_params(skl_linear_regression* m){

	PyObject* params = call_method(m->self, "get_params", NULL, NULL);
	if (params == NULL) {
		error("An error ocurred with scikitlearn!");
	}
	reprint(params);
	Py_DECREF(params);
}

static array* predict(skl_linear_regression* m, array* x){

	// converts array arguments to Python objects
	PyObject* PY_x = PyObject_from_double_array(x);

	// build arguments for predict with PY_x
	PyObject* args = build_arguments(1, PY_x);

	// call predict with PyObject* args and NULL kwargs
	// note: no need to call Py_DECREF(args) or Py_DECREF(kwargs)
	// call_method() takes care of those references
	PyObject* prediction = call_method(m->self, "predict", args, NULL);
	if (prediction == NULL) {
		error("An error ocurred when calling predict with x[]");
	}
	
	// note: there is no need to decrease reference count of PY_x
	// since build_arguments steals the reference from PY_x
	// thus and when we call call_method(), we do Py_DECREF(args),
	// which also decreases the reference count for PY_x

	if (!PyArray_Check(prediction)) {error("Not an array");}
	PyArrayObject* numpy_array = PyObject_to_PyArrayObject(prediction);
	
	array* arr = PyArrayObject_to_array(numpy_array);
	
	Py_DECREF(prediction);
	// no need to Py_DECREF(numpy_array), since numpy_array is the same
	// pointer, just cast to a PyArrayObject
	
	return arr;
}

// NOTE: doesn't support sample_weight
static float score(skl_linear_regression* m, array* x, array* y){

	// converts array arguments to Python objects
	PyObject* PY_x = PyObject_from_double_array(x);
	PyObject* PY_y = PyObject_from_double_array(y);
	
	// build arguments for score with PY_x and PY_y
	PyObject* args = build_arguments(2, PY_x, PY_y);
	
	// call score with PyObject* args and NULL kwargs
	// note: no need to call Py_DECREF(args) or Py_DECREF(kwargs)
	// call_method() takes care of those references
	PyObject* score = call_method(m->self, "score", args, NULL);
	
	if (score == NULL) {
		error("An error ocurred when calling score with x[] and/or y[]");
	}
	
	// note: there is no need to decrease reference count of PY_x,PY_y
	// since build_arguments steals the reference from PY_x,PY_y
	// thus and when we call call_method(), we do Py_DECREF(args),
	// which also decreases the reference count for PY_x,PY_y
	
	float score_c = float_from_PyObject(score);
	
	Py_DECREF(score);
	
	return score_c;
}

static void set_params(skl_linear_regression* m){
	
	// create Python dictionary of parameters
	PyObject *kwargs = PyDict_New();
	PyDict_SetItem(kwargs, PyUnicode_FromString("copy_X"), PyObject_from_boolean_int(m->parameters.copy_X));
    PyDict_SetItem(kwargs, PyUnicode_FromString("fit_intercept"), PyObject_from_boolean_int(m->parameters.fit_intercept));
    PyDict_SetItem(kwargs, PyUnicode_FromString("n_jobs"), PyObject_from_int(m->parameters.n_jobs));
    PyDict_SetItem(kwargs, PyUnicode_FromString("positive"), PyObject_from_boolean_int(m->parameters.positive)); 
    
	// call set_params with NULL args and kwargs
	// note: no need to call Py_DECREF(args) or Py_DECREF(kwargs)
	// call_method() takes care of those references
    PyObject *res = call_method(m->self, "set_params", NULL, kwargs);
    
	if (res == NULL) {
		error("An error ocurred while calling set_params!");
	}
	
	Py_DECREF(res);
}

static void purge(skl_linear_regression* m){

	// decreases reference count of LinearRegression Python instance
	Py_DECREF(m->self);
	
	// frees memory of skl_linear_regression struct
	free(m);
	
	// finalizes Python interpreter
	finalize_python();
}

static void get_parameter_defaults(skl_linear_regression* m){	
    m->parameters.fit_intercept = 1;
    m->parameters.copy_X		= 1;
    m->parameters.n_jobs		= 1;
    m->parameters.positive      = 0;
}

skl_linear_regression* skl_get_linear_regression(){

	// initialize Python interpreter
	initialize_python();
	
	// allocate memory for skl_linear_regression struct
	skl_linear_regression* m = malloc(sizeof(skl_linear_regression));
	
	// get Python instance of LinearRegression class from scikit-learn
	m->self = get_class_instance("sklearn.linear_model","LinearRegression", NULL);
	
	// get default parameters for skl_linear_regression
  	get_parameter_defaults(m);
  	
  	// setup function pointers for skl_linear_regression
  	m->fit 				 = &fit;
  	m->get_params 	     = &get_params;
  	m->predict 		     = &predict;
  	m->score 		     = &score;
  	m->set_params 		 = &set_params;
  	m->purge             = &purge;
  	return m;
}



