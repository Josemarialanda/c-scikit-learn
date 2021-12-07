# C bindings for scikit-learn

The c-scikit-learn package provides C bindings to scikit-learn.

This project is independent from [scikit-learn](https://scikit-learn.org/stable/).

# Documentation

[Go to documentation](https://github.com/Josemarialanda/C-wrapper-scikitlearn/blob/master/DOCUMENTATION.md).

# Examples

Toy example of a linear regression model ([full code](https://github.com/Josemarialanda/C-wrapper-scikitlearn/blob/master/examples/main.c))

```c
#include "skl/linear_model/linear_regression/linear_regression.h"

int main(){
    
    // get linear regression estimator
    linear_regression* reg = get_linear_regression();
    
    // declare training data size
    int r = 4;
    int c = 4;
    
    // get arrays for training data
    array* x = get_array(r,c);
    array* y = get_array(r,c);
  
  	// fill training data
    int count = 0;
    for (int i = 0; i < r; i++){
        for (int j = 0; j < c; j++){
            x->x[i][j] = ++count;
            y->x[i][j] = ++count+10;
        }
    }
            
    // fit with training data
    reg->fit(reg, x, y);
    
    // print some attributes after calling fit with (x,y)
	array* coef_ = reg->attributes.coef_;
	int    rank_ = reg->attributes.rank_;
	array* singular_ = reg->attributes.singular_;
	array* intercept_ = reg->attributes.intercept_;
	int    n_features_in_ = reg->attributes.n_features_in_;
	array* feature_names_in_ = reg->attributes.feature_names_in_;
	
	printf("Atributes:\n\n");
	
	printf("\tcoef_:\n");
	print_array(coef_);
	printf("\n\n");
	
	printf("\trank_:%i",rank_);
	printf("\n\n");

	printf("\tsingular_:\n");
	print_array(singular_);
	printf("\n\n");
	
	printf("\tintercept_:\n");
	print_array(intercept_);
	printf("\n\n");
	
	printf("\tn_features_in_:%i",n_features_in_);
	printf("\n\n");
    
    // score with training data
    double score = reg->score(reg, x, y);
    printf("Score: %f\n", score);
    
    // predict with testing data
    array* prediction = reg->predict(reg, x);
    printf("Prediction:\n");
    print_array(prediction);
   
    // free memory for created arrays
    free_array(x);
    free_array(y);
    free_array(prediction);
        
    // free memory for linear_regression
    reg->purge(reg);
    
    return 0;
}
```

# Installation Instructions

Note: building this repository with `nix` requires version 2.3 or newer. Check your stack version with `nix --version` in a terminal.

# Build on NixOS

The `shell.nix` provides an environment containing the necessary dependencies. To build, run:

```
$ nix-shell
```

This will enter the environment and build the project. Note, that it is an emulation of a common Linux
environment rather than the full-featured Nix package expression. No exportable Nix package will appear,
but local development is possible.

# Build with Docker on Linux

