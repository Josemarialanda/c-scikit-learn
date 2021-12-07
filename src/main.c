#include "skl/linear_model/linear_regression/linear_regression.h"

int main(){
    
    linear_regression* reg = get_linear_regression();
    
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
            
    reg->fit(reg, x, y);
    
    array* prediction = reg->predict(reg, x);
    
    printf("Prediction:\n");
    print_array(prediction);
   
    free_array(x);
    free_array(y);
    free_array(prediction); 
    reg->purge(reg);
    
    return 0;
}
