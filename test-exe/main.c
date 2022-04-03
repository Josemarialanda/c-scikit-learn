#include <c-skl/linear_model/linear_regression/skl_linear_regression.h>

array* fetch_data(int r, int c, int seed){
    array* x = skl_get_array(r,c);
  
    int count = 0;
    for (int i = 0; i < r; i++){
        for (int j = 0; j < c; j++){
            x->x[i][j] = ++count;
        }
    }
    return x;
}

int main(){
    
    initialize_skl();
    
    array* x = fetch_data(5,5, 25);
    array* y = fetch_data(5,5, 18);
  
    skl_linear_regression* reg = skl_get_linear_regression();
    reg->fit(reg, x, y);
    
    array* prediction = reg->predict(reg,x);
    printf("Prediction:\n\n");
    print_array(prediction);
   
    free_array(x);
    free_array(y);
    free_array(prediction);
    
    reg->purge(reg);
    
    finalize_skl();

    return 0;
}
