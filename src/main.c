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
    
    /*
    array* x = fetch_data(4,4, 25);
    array* y = fetch_data(4,4, 18);
  
    skl_linear_regression* reg = skl_get_linear_regression();
    reg->fit(reg, x, y);
    
    print_array(reg->attributes.singular_);
   
    free_array(x);
    free_array(y);
    
    reg->purge(reg);
    */

    array* x = fetch_data(4,4, 25);
    array* y = fetch_data(1,4, 18);
    
    skl_svr* svr = skl_get_svr();
    svr->fit(svr, x, y);
   
    free_array(x);
    free_array(y);
    
    svr->purge(svr);
	
    return 0;
}
