#include "helper.h"

void error(char* error){
	printf("%s\n", error);
	exit(1);
}

array* get_array(int rows, int cols){
    double** x = (double**)malloc((rows)*sizeof(double*));
    for (int i = 0; i < rows; i++){
        x[i] = (double*)malloc((cols)*sizeof(double));
    }
    array* arr = malloc(sizeof(array));
    arr->r = rows;
    arr->c = cols;
    arr->x = x;
    return arr;
}

void free_array(array* arr){
    for (int i = 0; i < arr->r; i++){
        free(arr->x[i]);
    }
    free(arr);
}

void initialize_python(){
	push_instance();
	if (is_instance_stack_full()){
		error("Python instance stack is full");
	}
	Py_Initialize();
	if(PyArray_API == NULL){
    	import_array(); 
	}
}

void finalize_python(){
	pop_instance();
	if (is_instance_stack_empty()){
		Py_Finalize();	
	}
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
	PyObject* class_instance = PyObject_CallObject(class, args);
	Py_DECREF(class);
	return class_instance;
}

PyObject* call_method(PyObject* class_instance, char* method_name, PyObject* args , PyObject* kwargs){

	if (args == NULL){
		PyObject *tuple = PyTuple_New(0);
		args = Py_BuildValue("O", tuple);	
	}

	PyObject* method = PyObject_GetAttrString(class_instance, method_name);
	if (method == NULL) {
		Py_DECREF(method);
		error("Can't find method");
	}
	PyObject* return_value = PyObject_Call(method, args, kwargs);
	Py_DECREF(method);
	if (return_value == NULL) {
		Py_DECREF(return_value);
		error("Incorrect method call");
	}
	return return_value;
}

PyObject* get_attribute(PyObject* class_instance, char* attribute_name){
	PyObject* attribute = PyObject_GetAttrString(class_instance, attribute_name);
	return attribute;
}

// not sure if I should call Py_DECREF somewhere ??? 
PyObject* build_arguments(int arg_count, ...){
	PyObject *tuple_args = PyTuple_New(arg_count);
	va_list args;
    va_start(args, arg_count);
    for (int i = 0; i < arg_count; i++){
      PyObject* arg = va_arg(args, PyObject*);
      PyTuple_SET_ITEM(tuple_args, i, arg);
    }
    va_end(args);
    // return Py_BuildValue("O", tuple_args);
    return tuple_args;
}


void reprint(PyObject *obj) {
    PyObject* repr = PyObject_Repr(obj);
    PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    const char *bytes = PyBytes_AS_STRING(str);

    printf("REPR: %s\n", bytes);

    Py_XDECREF(repr);
    Py_XDECREF(str);
}

PyObject* PyObject_from_string(char* s){
	return Py_BuildValue("s",s);
}

PyObject* PyObject_from_int(int n){
	return Py_BuildValue("i",n);
}

PyObject* PyObject_from_double(double d){
	return Py_BuildValue("d",d);
}

PyObject* PyObject_from_float(float f){
	return Py_BuildValue("f",f);
}

PyObject* PyObject_from_int_list(int* ns, size_t size){
	PyObject* list = PyList_New(size);
	for(int i=0; i<size;i++){
		PyObject* n = PyObject_from_int(ns[i]);
		PyList_SetItem(list, i, n);
	}
	return list;
}

PyObject* PyObject_from_double_list(double* ds, size_t size){
	PyObject* list = PyList_New(size);
	for(int i=0; i<size;i++){
		PyObject* d = PyObject_from_double(ds[i]);
		PyList_SetItem(list, i, d);
	}
	return list;
}

PyObject* PyObject_from_double_array(array* arr){
	PyObject* list = PyList_New(arr->r);
	for (int i=0; i < arr->r; i++){
	  	PyObject* list_i = PyObject_from_double_list(arr->x[i], arr->c);
	  	PyList_SetItem(list, i, list_i);
	}
	return list;
}

PyObject* PyObject_from_float_list(float* fs, size_t size){
	PyObject* list = PyList_New(size);
	for(int i=0; i<size;i++){
		PyObject* f = PyObject_from_float(fs[i]);
		PyList_SetItem(list, i, f);
	}
	return list;
}

PyObject* PyObject_from_boolean_int(int b){
	if (b != 1 && b != 0) { error("Not a truth value!"); }
	return PyBool_FromLong((long)b);
}

char* string_from_PyObject(PyObject* p_s){
	char* s;
	PyArg_Parse(p_s, "s", &s);
	return s;
}

int int_from_PyObject(PyObject* p_n){
	int n;
	PyArg_Parse(p_n, "i", &n);
	return n;
}

double double_from_PyObject(PyObject* p_d){
	double d;
	PyArg_Parse(p_d, "d", &d);
	return d;
}

float float_from_PyObject(PyObject* p_f){
	float f;
	PyArg_Parse(p_f, "f", &f);
	return f;
}

int* int_list_from_PyObject(PyObject* p_ns){
	int size = PyList_Size(p_ns);
	int* ns = malloc(sizeof(int)*size);
	for(int i=0; i<size;i++){
		ns[i] = int_from_PyObject(PyList_GetItem(p_ns, i));
	}
	return ns;
}

double* double_list_from_PyObject(PyObject* p_ds){
	int size = PyList_Size(p_ds);
	double* ds = malloc(sizeof(double)*size);
	for(int i=0; i<size;i++){
		ds[i] = double_from_PyObject(PyList_GetItem(p_ds, i));
	}
	return ds;
}

float* float_list_from_PyObject(PyObject* p_fs){
	int size = PyList_Size(p_fs);
	float* fs = malloc(sizeof(float)*size);
	for(int i=0; i<size;i++){
		fs[i] = float_from_PyObject(PyList_GetItem(p_fs, i));
	}
	return fs;
}

int boolean_int_from_from_PyObject(PyObject* b){
	if (!PyBool_Check(b)) { error("PyObject is not of type PyBool_Type"); }
	return PyObject_IsTrue(b);
}

PyArrayObject* PyObject_to_PyArrayObject(PyObject* a){
	PyArrayObject* numpy_array = (PyArrayObject*) a;
	return numpy_array;
}







