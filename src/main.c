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
    
    // predict with testing data
    reg->predict(reg, x);
    
    // score with training data
    // double score = reg->score(reg, x, y);
    
    // free memory for arrays
    free_array(x);
    free_array(y);
        
    // free memory for linear_regression
    reg->purge(reg);
    
    return 0;
}
