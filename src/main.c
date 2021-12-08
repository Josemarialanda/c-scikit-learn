#include "skl/linear_model/linear_regression/skl_linear_regression.h"
#include "skl/svm/svr/skl_svr.h"

array* fetch_data(int r, int c, int seed){
    array* x = get_array(r,c);
  
    int count = 0;
    for (int i = 0; i < r; i++){
        for (int j = 0; j < c; j++){
            x->x[i][j] = ++count; // ++count*seed*(1/25);
        }
    }
    return x;
}

int main(){
    
    array* x = fetch_data(4,4, 25);
    array* y = fetch_data(4,4, 18);
  
    skl_linear_regression* reg = skl_get_linear_regression();
    reg->fit(reg, x, y);
    printf("Score: %f", reg->score(reg, x, y));
    
    skl_svr* svr = skl_get_svr();
   
    free_array(x);
    free_array(y);
    
    svr->purge(svr);
    reg->purge(reg);
    
    return 0;
}
