#define PY_SSIZE_T_CLEAN
#include <Python.h>

// TODO: Add meaningful error messages
void error(char* error){
	printf("%s\n", error);
	exit(1);
}

PyObject* get_module(char* module_name){
	return PyImport_ImportModule(module_name);;
}

PyObject* get_class(PyObject* module, char* class_name){
	return PyObject_GetAttrString(module, class_name);
}

PyObject* get_class_instance(char* module_name, char* class_name, PyObject* args){
	PyObject* module = get_module(module_name);
	if (module == NULL) {
		Py_DECREF(module);
		error("Can't find module");
	}
	PyObject* class  = get_class(module, class_name);
	Py_DECREF(module);
	if (class == NULL) {
		Py_DECREF(class);
		error("Can't find class");
	}
	PyObject* class_instance = PyEval_CallObject(class, args);
	Py_DECREF(class);
	return class_instance;
}

PyObject* call_method(PyObject* class_instance, char* method_name, PyObject* args){
	PyObject* method = PyObject_GetAttrString(class_instance, method_name);
	if (method == NULL) {
		Py_DECREF(method);
		error("Can't find method");
	}
	PyObject* return_value = PyEval_CallObject(method, args);
	Py_DECREF(method);
	if (return_value == NULL) {
		Py_DECREF(return_value);
		error("Incorrect method call");
	}
	return return_value;
}

void reprint(PyObject *obj) {
    PyObject* repr = PyObject_Repr(obj);
    PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    const char *bytes = PyBytes_AS_STRING(str);

    printf("REPR: %s\n", bytes);

    Py_XDECREF(repr);
    Py_XDECREF(str);
}

int main(){

	/*

	Py_Initialize();
	
	// load LinearRegression from scikit-learn
	PyObject* LinearRegression = get_class_instance("sklearn.linear_model","LinearRegression", NULL);
	
	// call get_params method in LinearRegression
	PyObject* return_value = call_method(LinearRegression, "get_params", NULL);
	
	// decrease reference to LinearRegression object
	Py_DECREF(LinearRegression);
	
	// parse params dictionary
	
	// dictionary keys
	PyObject *py_copy_X_key = PyUnicode_FromString("copy_X");
	PyObject *py_fit_intercept_key = PyUnicode_FromString("fit_intercept");
	PyObject *py_n_jobs_key = PyUnicode_FromString("n_jobs");
	PyObject *py_normalize_key = PyUnicode_FromString("normalize");
	PyObject *py_positive_key = PyUnicode_FromString("positive");
	
	// dictionary values
	PyObject *py_copy_X = PyDict_GetItem(return_value, py_copy_X_key);
	PyObject *py_fit_intercept = PyDict_GetItem(return_value, py_fit_intercept_key);
	PyObject *py_n_jobs = PyDict_GetItem(return_value, py_n_jobs_key);
	PyObject *py_normalize = PyDict_GetItem(return_value, py_normalize_key);
	PyObject *py_positive = PyDict_GetItem(return_value, py_positive_key);
	
	int copy_X = PyObject_IsTrue(py_copy_X);
	int fit_intercept = PyObject_IsTrue(py_fit_intercept);
	int n_jobs = PyObject_IsTrue(py_n_jobs);
	int normalize = PyObject_IsTrue(py_normalize);
	int positive = PyObject_IsTrue(py_positive);
	
	printf("copy_X = %d\n", copy_X);
	printf("fit_intercept = %d\n", fit_intercept);
	printf("n_jobs = %d\n", n_jobs);
	printf("normalize = %d\n", normalize);
	printf("positive = %d\n", positive);
	
	*/
	
	Py_Initialize();

    // convert C arrays to python objects
    double X[4][2] = {{1, 1}, {1, 2}, {2, 2}, {2, 3}};
    double y[4]    = {2, 4, 6, 9};
    
    PyObject* PY_X = PyList_New(0);
	for (int i=0; i<4; i++){
		PyObject *list  = PyList_New(2);
		for(int j = 0; j < 2; j++){
			PyList_SetItem(list,j, PyFloat_FromDouble(X[i][j]));
		}
		PyList_Append(PY_X, list);
	}
	
	PyObject* PY_y = PyList_New(0);
	for(int i=0; i<4;i++){
		PyObject* d = PyFloat_FromDouble(y[i]);
		PyList_Append(PY_y, d);
	}
	
	
	assert(PY_X != NULL);
	assert(PY_y != NULL);
	
	// build arguments for fit method
	PyObject *tuple_fit = PyTuple_New(2); // 2 arguments
	PyTuple_SET_ITEM(tuple_fit, 0, PY_X);
	PyTuple_SET_ITEM(tuple_fit, 1, PY_y);
	PyObject *args_fit = Py_BuildValue("O", tuple_fit);	

	// load LinearRegression from scikit-learn
	PyObject* LinearRegression = get_class_instance("sklearn.linear_model","LinearRegression", NULL);
	
	// call get_params method in LinearRegression
	PyObject* fitted_regressor = call_method(LinearRegression, "fit", args_fit);
	
	if (fitted_regressor == NULL) {
		Py_DECREF(fitted_regressor);
		error("Incorrect method call");
	}
	
	// build arguments for predict method
	PyObject *tuple_predict = PyTuple_New(1); // 2 arguments
	PyTuple_SET_ITEM(tuple_predict, 0, PY_X);
	PyObject *args_predict = Py_BuildValue("O", tuple_predict);	
	
	// call predic method in LinearRegression
	// you can call predict with either LinearRegression or fitted_regressor
	PyObject* prediction = call_method(fitted_regressor, "predict", args_predict);
	
	reprint(prediction);
	
	// decrease reference to fitted_regressor object
	Py_DECREF(fitted_regressor);
	
	// decrease reference to LinearRegression object
	Py_DECREF(LinearRegression);
	
	Py_Finalize();

	return 0;
}
