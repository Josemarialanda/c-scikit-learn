#include "skl_helper.h"

void print_array(array* arr){
	int r = arr->r;
	int c = arr->c;
	double** x = arr->x;
	for (int i = 0; i < r; ++i){
		printf("arr[%i]: ",i);
		for (int j = 0; j < c; ++j) {
			printf("%lf ", x[i][j]);
		}
		printf("\n");
	}
}

// TODO: Improve error messages
void error(char* error){
	printf("%s\n", error);
	exit(1);
}

array* skl_get_array(int rows, int cols){
    double** x = (double**)malloc((rows)*sizeof(double*));
    for (int i = 0; i < rows; i++){
        x[i] = (double*)malloc((cols)*sizeof(double));
    }
    
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            x[i][j] = 0;
        }
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

void initialize_skl(){
	Py_Initialize();	
	if(PyArray_API == NULL){ import_array();}
}

void finalize_skl(){
	Py_Finalize();
}

PyObject* get_module(char* module_name){
	// increases reference -> requires call to PY_DECREF()
	// might return NULL, in that case the refrence count is not increased
	return PyImport_ImportModule(module_name);
}

PyObject* get_class(PyObject* module, char* class_name){
	// increases reference -> requires call to PY_DECREF()
	// might return NULL, in that case the refrence count is not increased
	return PyObject_GetAttrString(module, class_name);
}

PyObject* get_class_instance(char* module_name, char* class_name, PyObject* args){

	PyObject* module = get_module(module_name);
	if (module == NULL) {
		// no need to decrease reference count on PyObject* module
		error("Can't find module");
	}
	PyObject* class  = get_class(module, class_name);
	Py_DECREF(module);
	if (class == NULL) {
		// no need to decrease reference count on PyObject* class
		error("Can't find class");
	}
	PyObject* class_instance = PyObject_CallObject(class, args);
	// decrease reference count on PyObject* args (could be NULL)
	Py_XDECREF(args);
	// decrease reference count on PyObject* class
	Py_DECREF(class);
	return class_instance;
}

PyObject* call_method(PyObject* class_instance, char* method_name, PyObject* args , PyObject* kwargs){

	// note we don't call PY_DECREF() on class_instance,
	// reference count of class_instance is managed by 
	// the model code

	if (args == NULL){
		PyObject *tuple = PyTuple_New(0);
		args = Py_BuildValue("O", tuple);	
	}

	PyObject* method = PyObject_GetAttrString(class_instance, method_name);
	if (method == NULL) {
		error("Can't find method");
	}
	
	PyObject* return_value = PyObject_Call(method, args, kwargs);
	Py_DECREF(method);
	if (return_value == NULL) {
		error("Incorrect method call");
	}
	// decrease reference count on PyObject* args
	Py_DECREF(args);
	// decrease reference count on PyObject* kwargs (could be NULL)
	Py_XDECREF(kwargs);
	return return_value;
}

PyObject* get_attribute(PyObject* class_instance, char* attribute_name){ 

	// note we don't call Py_DECREF() on class_instance,
	// reference count of class_instance is managed by 
	// the model code

	PyObject* attribute;
	if (!PyObject_HasAttrString(class_instance, attribute_name)){
		return NULL;
	}
	attribute = PyObject_GetAttrString(class_instance, attribute_name);
	return attribute;
}

PyObject* build_arguments(int arg_count, ...){
	PyObject *tuple_args = PyTuple_New(arg_count);
	va_list args;
    va_start(args, arg_count);
    for (int i = 0; i < arg_count; i++){
      PyObject* arg = va_arg(args, PyObject*);
      PyTuple_SET_ITEM(tuple_args, i, arg);
      // PyTuple_SET_ITEM steals a reference to arg
      // thus, there is no need to call Py_DECREF(arg);
    }
    va_end(args);
    return tuple_args;
}

void reprint(PyObject *obj) {
	// doesnt't manage references to obj
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
        // PyList_SetItem steals a reference to n
        // thus, there is no need to call Py_DECREF(n);
	}
	return list;
}

PyObject* PyObject_from_double_list(double* ds, size_t size){
	PyObject* list = PyList_New(size);
	for(int i=0; i<size;i++){
		PyObject* d = PyObject_from_double(ds[i]);
		PyList_SetItem(list, i, d);
        // PyList_SetItem steals a reference to d
        // thus, there is no need to call Py_DECREF(d);
	}
	return list;
}

PyObject* PyObject_from_double_array(array* arr){
	PyObject* list = PyList_New(arr->r);
	for (int i=0; i < arr->r; i++){
	  	PyObject* list_i = PyObject_from_double_list(arr->x[i], arr->c);
	  	PyList_SetItem(list, i, list_i);
        // PyList_SetItem steals a reference to i
        // thus, there is no need to call Py_DECREF(i);
	}
	return list;
}

PyObject* PyObject_from_float_list(float* fs, size_t size){
	PyObject* list = PyList_New(size);
	for(int i=0; i<size;i++){
		PyObject* f = PyObject_from_float(fs[i]);
		PyList_SetItem(list, i, f);
        // PyList_SetItem steals a reference to f
        // thus, there is no need to call Py_DECREF(f);
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
        // PyList_GetItem steals a reference to p_ns
        // thus, there is no need to call Py_DECREF(p_ns);
		ns[i] = int_from_PyObject(PyList_GetItem(p_ns, i));
	}
	return ns;
}

double* double_list_from_PyObject(PyObject* p_ds){
	int size = PyList_Size(p_ds);
	double* ds = malloc(sizeof(double)*size);
	for(int i=0; i<size;i++){
        // PyList_GetItem steals a reference to p_ds
        // thus, there is no need to call Py_DECREF(p_ds);
		ds[i] = double_from_PyObject(PyList_GetItem(p_ds, i));
	}
	return ds;
}

float* float_list_from_PyObject(PyObject* p_fs){
	int size = PyList_Size(p_fs);
	float* fs = malloc(sizeof(float)*size);
	for(int i=0; i<size;i++){
        // PyList_GetItem steals a reference to p_fs
        // thus, there is no need to call Py_DECREF(p_fs);
		fs[i] = float_from_PyObject(PyList_GetItem(p_fs, i));
		
	}
	return fs;
}

int boolean_int_from_from_PyObject(PyObject* b){
	if (!PyBool_Check(b)) { error("PyObject is not of type PyBool_Type"); }
	return PyObject_IsTrue(b);
}

array* PyObject_tuple_to_array(PyObject* t){
    int len = PyTuple_Size(t);
    array* arr = skl_get_array(1,len);
    for(int i = 0; i < len; ++i){
		PyObject* obj = PyTuple_GetItem(t,i);
		arr->x[0][i] = double_from_PyObject(obj);
		Py_DECREF(obj);
	}
	return arr;
}

PyArrayObject* PyObject_to_PyArrayObject(PyObject* a){
	PyArrayObject* numpy_array = (PyArrayObject*) a;
	return numpy_array;
}

array* PyArrayObject_to_array(PyArrayObject* a){
	// extract dimensions from numpy array
    int dims = PyArray_NDIM(a);
    // declare array struct
    array* arr;
	if (dims == 2){
		// extract rows info from numpy array
	    npy_intp   r = PyArray_DIMS(a)[0];
	    // extract columns info from numpy array
    	npy_intp   c = PyArray_DIMS(a)[1];
    	// create array with correct dimensions
    	arr = skl_get_array(r,c);
    	// extract datatype from numpy array
    	int typenum  = PyArray_TYPE(a);
    	// parse array depending on internal datatype
		switch(typenum) {
			case NPY_FLOAT:
				for (int i = 0; i < r; ++i){
					for (int j = 0; j < c; ++j) {
						arr->x[i][j] = *(float*)PyArray_GetPtr(a, (npy_intp[]){i, j} );
					}
				}
				break; 
			case NPY_DOUBLE:
				for (int i = 0; i < r; ++i){
					for (int j = 0; j < c; ++j) {
						arr->x[i][j] = *(double*)PyArray_GetPtr(a, (npy_intp[]){i, j} );
					}
				}
				break;
			case NPY_INT32:
				for (int i = 0; i < r; ++i){
					for (int j = 0; j < c; ++j) {
						arr->x[i][j] = *(int*)PyArray_GetPtr(a, (npy_intp[]){i, j} );
					}
				}
				break;
			default:
				error("unknown datatype");
		}

	} else {
		// extract columns info from numpy array
      	npy_intp   c = PyArray_DIMS(a)[0];
      	// create array with correct dimensions
      	arr = skl_get_array(1,c);
      	// extract datatype from numpy array
    	int typenum  = PyArray_TYPE(a);
    	// parse array depending on internal datatype
		switch(typenum) {
			case NPY_FLOAT:
				for (int i = 0; i < c; ++i) {
					arr->x[0][i] = *(float*)PyArray_GetPtr(a, (npy_intp[]){i} );
				}
				break; 
			case NPY_DOUBLE:
				for (int i = 0; i < c; ++i) {
					arr->x[0][i] = *(double*)PyArray_GetPtr(a, (npy_intp[]){i} );
				}
				break;
			case NPY_INT32:
				for (int i = 0; i < c; ++i) {
					arr->x[0][i] = *(int*)PyArray_GetPtr(a, (npy_intp[]){i} );
				}
				break;
			default:
				error("unknown datatype");
		}
	}
	return arr;
}







