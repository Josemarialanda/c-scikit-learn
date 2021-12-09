#ifndef SKL_LINEAR_REGRESSION
#define SKL_LINEAR_REGRESSION

#define NO_IMPORT_ARRAY
#include "../../skl_helper.h"

typedef struct skl_linear_regression skl_linear_regression;

// skl_linear_regression model struct
struct skl_linear_regression {
	// scikit-learn linearRegression class instance
	PyObject* self;
	struct {		
		int fit_intercept;
		int copy_X;
		int n_jobs;
		int positive;	
	} parameters;
	struct {
		array* coef_;
		int      rank_;
		array*  singular_;
		array*  intercept_;
		int      n_features_in_;
		array*  feature_names_in_;
	} attributes;
	// methods
 	void (*fit)(skl_linear_regression* m, array* x, array* y);
 	void (*get_params)(skl_linear_regression* m);
 	array* (*predict)(skl_linear_regression* m, array* x);
 	float (*score)(skl_linear_regression* m, array* x, array* y);
 	void (*set_params)(skl_linear_regression* m);
 	void (*purge)(skl_linear_regression* m);
};

// get skl_linear_regression model
skl_linear_regression* skl_get_linear_regression();

#endif /* SKL_LINEAR_REGRESSION */
