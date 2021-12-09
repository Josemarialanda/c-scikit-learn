# Scikit-learn API Reference

To learn more about the scikit-learn API [click here](https://scikit-learn.org/stable/modules/classes.html#).


# c-scikit-learn implementation


The folder structure of [c-scikit-learn](https://github.com/Josemarialanda/c-scikit-learn) is the following:

```
src
│   Makefile
│   shell.nix
│
└───skl
│   │   helper.h
│   │   helper.c
│   │
│   └───linear_model/linear_regression
│       │   skl_linear_regression.h
│       │   skl_linear_regression.c
│       │   ...
│
└───svm/svr
    │   skl_svr.h
    │   skl_svr.c
```

Each folder in skl contains a distinct model from [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#).

# array datatype

[c-scikit-learn](https://github.com/Josemarialanda/c-scikit-learn) makes use of a `double**` wrapper, which is used by every model in [c-scikit-learn](https://github.com/Josemarialanda/c-scikit-learn).

```c
typedef struct array{
  int r;
  int c;
  double** x;
} array;
```

# Model structure

A model has the following structure:

```c
#ifndef SKL_<MODEL_NAME>
#define SKL_<MODEL_NAME>

#define NO_IMPORT_ARRAY
#include "../../helper.h"

typedef struct skl_<model_name> skl_<model_name>;

// skl_<model_name> model struct
struct skl_<model_name> {
	PyObject* self; // reference to a scikit-learn object
	struct {
	   int parameter_1;
	   int parameter_2;
	   .
	   .
	   .
	   char* parameter_n;
	} parameters;
	struct {
	   int attribute_1;
	   int attribute_2;
	   .
	   .
	   .
	   array* attribute_3;
	} attributes;
	// methods associated to <model_name>
 	void (*fit)(skl_<model_name>* m, array* x, array* y);
 	void (*get_params)(skl_<model_name>* m);
 	array* (*predict)(skl_<model_name>* m, array* x);
 	float (*score)(skl_<model_name>* m, array* x, array* y);
 	void (*set_params)(skl_<model_name>* m);
 	void (*purge)(skl_<model_name>* m);
};

// fetches a <model_name> struct with default parameters
skl_<model_name>* skl_get_<model_name>();

#endif /* SKL_<MODEL_NAME> */
```

Other than that, the functionality is pretty much the same as in [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#).

For an example, check out [toy linear regression example](https://github.com/Josemarialanda/c-scikit-learn/blob/master/examples/main.c).