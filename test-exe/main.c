#include <c-skl/linear_model/linear_regression/skl_linear_regression.h>
#include <c-skl/svm/svr/skl_svr.h>
#include <c-skl/skl_helper.h>


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
    
    initialize_skl();
    
    // linear regression
    array* x1 = fetch_data(4,4, 25);
    array* y1 = fetch_data(4,4, 18);
  
    skl_linear_regression* reg = skl_get_linear_regression();
    reg->fit(reg, x1, y1);
    
    printf("Score: %f\n", reg->score(reg,x1,y1));
   
    free_array(x1);
    free_array(y1);
    
    reg->purge(reg);
    
	// support vector machine regression
    array* x2 = fetch_data(4,4, 25);
    array* y2 = fetch_data(1,4, 18);
    
    skl_svr* svr = skl_get_svr();
    svr->fit(svr, x2, y2);

    printf("Score = %f\n", svr->score(svr,x2,y2));
   
    free_array(x2);
    free_array(y2);

    svr->purge(svr);
    
    finalize_skl();

    return 0;
}
