#ifndef HELPER
#define HELPER

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL HELPER_ARRAY_API
#include <numpy/arrayobject.h>
#include "stack.h"

// Wrapper for a double multi dimensional array
typedef struct array{
  int r;
  int c;
  double** x;
} array;

void print_array(array* arr);

// report errors
void error(char* error);

// Gets a pointer to an array struct
array* get_array(int rows, int cols);

// Frees the memory of an array struct
void free_array(array* arr);




// Python-C API helper functions

// initialiize the Python interpreter
void initialize_python();

// finalize the Python interpreter
void finalize_python();

// get Python module
PyObject* get_module(char* module_name);

// get class in Python module
PyObject* get_class(PyObject* module, char* class_name);

// get class instance of class
PyObject* get_class_instance(char* module_name, char* class_name, PyObject* args);

// call method in class instance
PyObject* call_method(PyObject* class_instance, char* method_name, PyObject* args, PyObject* kwargs);

// Get attribute from class instance
PyObject* get_attribute(PyObject* class_instance, char* attribute_name);

// Build arguments for function call
PyObject* build_arguments(int arg_count, ...);

// call __repr__() on Python Python object
void reprint(PyObject *obj);

// convert a null-terminated C string to a Python object
// if the C string pointer is NULL, None is returned
PyObject* PyObject_from_string(char* s);

// convert an integer to a Python object
PyObject* PyObject_from_int(int n);

// convert a double to a Python object
PyObject* PyObject_from_double(double d);

// convert a float to a Python object
PyObject* PyObject_from_float(float f);

// convert a int array to a Python object
PyObject* PyObject_from_int_list(int* ns, size_t size);

// convert a double array to a Python object
PyObject* PyObject_from_double_list(double* ds, size_t size);

// convert an array datatype matrix to a Python object
PyObject* PyObject_from_double_array(array* arr);

// convert a float array to a Python object
PyObject* PyObject_from_float_list(float* fs, size_t size);

// convert an int (boolean) a Python object
PyObject* PyObject_from_boolean_int(int b);

// convert a Python object to a null-terminated C string
// if the Python object is None, NULL is returned
char* string_from_PyObject(PyObject* p_s);

// convert Python object to an integer
int int_from_PyObject(PyObject* p_n);

// convert Python object to a double
double double_from_PyObject(PyObject* p_d);

// convert Python object to a float
float float_from_PyObject(PyObject* p_);

// convert Python object to an integer array
int* int_list_from_PyObject(PyObject* p_ns);

// convert Python object to a double array
double* double_list_from_PyObject(PyObject* p_ds);

// convert Python object to a float array
float* float_list_from_PyObject(PyObject* p_fs);

// convert Python object to an int (boolean)
int boolean_int_from_from_PyObject(PyObject* b);





// Numpy helper functions

// convert a PyObject to a PyArrayObject (numpy array)
PyArrayObject* PyObject_to_PyArrayObject(PyObject* a);

// convert a PyArrayObject to an array struct
array* PyArrayObject_to_array(PyArrayObject* a);

#endif /* HELPER */














