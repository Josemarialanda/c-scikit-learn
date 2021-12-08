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
	PyObject* shape_fit_ = get_attribute(m->self, "shape_fit_");
	PyArrayObject* support_ = PyObject_to_PyArrayObject(get_attribute(m->self, "support_"));
	PyArrayObject* support_vectors_ = PyObject_to_PyArrayObject(get_attribute(m->self, "support_vectors_"));
	// always defined
	m->attributes.class_weight_ = PyArrayObject_to_array(class_weight_);
	m->attributes.dual_coef_ = PyArrayObject_to_array(dual_coef_);
	m->attributes.fit_status_ = int_from_PyObject(fit_status_);
	m->attributes.intercept_ = PyArrayObject_to_array(intercept_);
	m->attributes.n_features_in_ = int_from_PyObject(n_features_in_);
	m->attributes.n_support_ = PyArrayObject_to_array(n_support_);
	m->attributes.shape_fit_ = PyObject_tuple_to_array(shape_fit_); // TODO: convert tuple to array
	m->attributes.support_ = PyArrayObject_to_array(support_);
	m->attributes.support_vectors_ = PyArrayObject_to_array(support_vectors_);
	// may be NULL
	m->attributes.coef_ = NULL;
	m->attributes.feature_names_in_ = NULL;
	// defined only when X has feature names that are all strings.
	if (coef_ != NULL) { m->attributes.coef_ = PyArrayObject_to_array(coef_); }
	if (feature_names_in_ != NULL) { m->attributes.feature_names_in_ = PyArrayObject_to_array(feature_names_in_); }
	Py_DECREF(class_weight_);
	Py_DECREF(dual_coef_);
	Py_DECREF(fit_status_);
	Py_DECREF(intercept_);
	Py_DECREF(n_features_in_);
	Py_DECREF(n_support_);
	Py_DECREF(shape_fit_);
	Py_DECREF(support_);
	Py_DECREF(support_vectors_);
	Py_XDECREF(coef_);
	Py_XDECREF(feature_names_in_);
	Py_DECREF(fitted_estimator);
}

static void get_params(skl_svr* m){
	PyObject* params = call_method(m->self, "get_params", NULL, NULL);
	if (params == NULL) {
		error("An error ocurred with scikitlearn!");
	}
	reprint(params);
	Py_DECREF(params);
}

static array* predict(skl_svr* m, array* x){
	PyObject* PY_x = PyObject_from_double_array(x);
	PyObject* args = build_arguments(1, PY_x);
	PyObject* prediction = call_method(m->self, "predict", args, NULL);
	if (prediction == NULL) {
		error("An error ocurred when calling predict with x[]");
	}
	if (!PyArray_Check(prediction)) {error("Not an array");}
	PyArrayObject* numpy_array = PyObject_to_PyArrayObject(prediction);
	array* arr = PyArrayObject_to_array(numpy_array);
	Py_DECREF(prediction);
	return arr;
}

static float score(skl_svr* m, array* x, array* y){
	PyObject* PY_x = PyObject_from_double_array(x);
	PyObject* PY_y = PyObject_from_double_list(y->x[0], y->c);
	PyObject* args = build_arguments(2, PY_x, PY_y);
	PyObject* score = call_method(m->self, "score", args, NULL);
	if (score == NULL) {
		error("An error ocurred when calling score with x[] and/or y[]");
	}
	float score_c = float_from_PyObject(score);
	Py_DECREF(score);
	return score_c;
}

static void set_params(skl_svr* m){
	PyObject* kwargs = PyDict_New();
	PyDict_SetItem(kwargs, PyUnicode_FromString("kernel"), PyObject_from_string(m->parameters.kernel));
	PyDict_SetItem(kwargs, PyUnicode_FromString("degree"), PyObject_from_int(m->parameters.degree));
	PyDict_SetItem(kwargs, PyUnicode_FromString("gamma"), PyObject_from_string(m->parameters.gamma));
	PyDict_SetItem(kwargs, PyUnicode_FromString("coef0"), PyObject_from_float(m->parameters.coef0));
	PyDict_SetItem(kwargs, PyUnicode_FromString("tol"), PyObject_from_float(m->parameters.tol));
	PyDict_SetItem(kwargs, PyUnicode_FromString("C"), PyObject_from_float(m->parameters.C));
	PyDict_SetItem(kwargs, PyUnicode_FromString("epsilon"), PyObject_from_float(m->parameters.epsilon));
	PyDict_SetItem(kwargs, PyUnicode_FromString("shrinking"), PyObject_from_int(m->parameters.shrinking));
	PyDict_SetItem(kwargs, PyUnicode_FromString("cache_size"), PyObject_from_float(m->parameters.cache_size));
	PyDict_SetItem(kwargs, PyUnicode_FromString("verbose"), PyObject_from_boolean_int(m->parameters.verbose));
	PyDict_SetItem(kwargs, PyUnicode_FromString("max_iter"), PyObject_from_int(m->parameters.max_iter)); 
    PyObject *res = call_method(m->self, "set_params", NULL, kwargs);
	if (res == NULL) {
		error("An error ocurred while calling set_params!");
	}
	Py_DECREF(res);
}

static void purge(skl_svr* m){
	Py_DECREF(m->self);
	free(m);
}

static void get_parameter_defaults(skl_svr* m){	
    m->parameters.kernel     = "rbf";
    m->parameters.degree     = 3;
    m->parameters.gamma      = "scale";
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



