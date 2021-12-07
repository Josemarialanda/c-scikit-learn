# C bindings for scikit-learn

The c-scikit-learn package provides C bindings to scikit-learn.

This project is independent from [scikit-learn](https://scikit-learn.org/stable/).

# Documentation

[Go to documentation](https://github.com/Josemarialanda/C-wrapper-scikitlearn/blob/master/DOCUMENTATION.md).

# Examples

Toy example of a linear regression model ([full code](https://github.com/Josemarialanda/C-wrapper-scikitlearn/blob/master/examples/main.c))

```c
#include "skl/linear_model/linear_regression/skl_linear_regression.h"

int main(){
    
    skl_linear_regression* reg = skl_get_linear_regression();
    
    int r = 4;
    int c = 4;
    
    array* x = get_array(r,c);
    array* y = get_array(r,c);
  
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
```

```
./out

Prediction:
arr[0]: 1.000000 3.000000 5.000000 7.000000
arr[1]: 9.000000 11.000000 13.000000 15.000000
arr[2]: 17.000000 19.000000 21.000000 23.000000
arr[3]: 25.000000 27.000000 29.000000 31.000000
```

# Dependencies

The project was built with:

* GNU Make 4.3
* GCC 10.3.0
* Nix 2.3
* Python 3.9.6
* Scikit-learn 1.0.1
* Numpy 1.21.4

# Installation Instructions

Note: building this repository with `nix` requires version 2.3 or newer. Check your nix version with `nix --version` in a terminal.

# Build on NixOS

The `shell.nix` provides an environment containing the necessary dependencies. To enter the build environment, run:

```
$ nix-shell
```

To build, run (from within the shell):

```
$ make
```

This will enter the environment and build the project. Note, that it is an emulation of a common Linux
environment rather than the full-featured Nix package expression. No exportable Nix package will appear,
but local development is possible.

# Build with Docker on Linux
