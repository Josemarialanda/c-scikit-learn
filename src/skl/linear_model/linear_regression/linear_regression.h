#define NO_IMPORT_ARRAY
#include "../../helper.h"

typedef struct linear_regression linear_regression;

// linear_regression model struct
struct linear_regression {
	// python linear_regression class instance
	PyObject* self;
	struct {		
		int fit_intercept;
		int copy_X;
		int n_jobs;
		int positive;	
	} parameters;
	struct {
		double** coef_;
		int      rank_;
		double*  singular_;
		double*  intercept_;
		int      n_features_in_;
		double*  feature_names_in_;
	} attributes;
	// methods
 	void (*fit)(linear_regression* m, array* x, array* y);
 	void (*get_params)(linear_regression* m);
 	array* (*predict)(linear_regression* m, array* x);
 	double (*score)(linear_regression* m, array* x, array* y);
 	void (*set_params)(linear_regression* m);
 	void (*purge)(linear_regression* m);
};

// get linear_regression model
linear_regression* get_linear_regression();













