#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "skl/linear_model/linear_regression/linear_regression.h"

int main(){
    
    // get linear regression estimator
    linear_regression* reg = get_linear_regression();
    
    // declare training data array size
    int r = 4;
    int c = 4;
    
    // get arrays for training data
    array* x = get_array(r,c);
    array* y = get_array(r,c);
  
  	// fill training data (example)
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
	
	// feature_names_in_ is not defined for x,y
    
    // score with training data
    double score = reg->score(reg, x, y);
    printf("Score: %f\n", score);
    
    // predict with testing data
    array* prediction = reg->predict(reg, x);
    printf("Prediction:\n");
    print_array(prediction);
   
    // free memory for arrays
    free_array(x);
    free_array(y);
    free_array(prediction);
        
    // free memory for linear_regression
    reg->purge(reg);
    
    return 0;
}
