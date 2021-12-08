#define NO_IMPORT_ARRAY
#include "../../helper.h"

typedef struct skl_svr skl_svr;

// skl_svr model struct
struct skl_svr {
	// scikit-learn SVR class instance
	PyObject* self;
	struct {		
		char* kernel;
		int   degree;
		char* gamma;
		float coef0;
		float tol;
		float C;
		float epsilon;
		int   shrinking;
		float cache_size;
		int   verbose;
		int   max_iter;
	} parameters;
	struct {
		array* class_weight_;
		array* coef_;
		array* dual_coef_;
		int    fit_status_;
		array* intercept_;
		int    n_features_in_;
		array* feature_names_in_;
		array* n_support_;
		array* shape_fit_;
		array* support_;
		array* support_vectors_;
	} attributes;
	// methods
 	void (*fit)(skl_svr* m, array* x, array* y);
 	void (*get_params)(skl_svr* m);
 	array* (*predict)(skl_svr* m, array* x);
 	double (*score)(skl_svr* m, array* x, array* y);
 	void (*set_params)(skl_svr* m);
 	void (*purge)(skl_svr* m);
};

// get skl_svr model
skl_svr* skl_get_svr();













