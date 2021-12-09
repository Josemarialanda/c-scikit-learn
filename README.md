# C bindings for scikit-learn

The c-scikit-learn package provides C bindings to scikit-learn.

This project is independent from [scikit-learn](https://scikit-learn.org/stable/).

# Documentation

[Go to documentation](https://github.com/Josemarialanda/C-wrapper-scikitlearn/blob/master/DOCUMENTATION.md).

# Examples

Toy example of a linear regression model ([full code](https://github.com/Josemarialanda/c-scikit-learn/blob/master/test-exe/main.c))

```c
#include <c-skl/linear_model/linear_regression/skl_linear_regression.h>

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

    array* x = fetch_data(4,4, 25);
    array* y = fetch_data(4,4, 18);

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
```

```
./main-exe
Prediction:

arr[0]: 1.000000 2.000000 3.000000 4.000000
arr[1]: 5.000000 6.000000 7.000000 8.000000
arr[2]: 9.000000 10.000000 11.000000 12.000000
arr[3]: 13.000000 14.000000 15.000000 16.000000
```

# Dependencies

[c-scikit-learn](https://github.com/Josemarialanda/c-scikit-learn) was built with:

* GNU Make 4.3
* CMAKE 3.21.2
* GCC 10.3.0
* Nix 2.3
* Python 3.9.6
* Scikit-learn 1.0.1
* Numpy 1.21.4

# Installation Instructions

Note: building this repository with `nix` requires version 2.3 or newer. Check your nix version with `nix --version` in a terminal.

# Build with Nix

## Install Nix

The quickest way to install Nix is to open a terminal and run the following command (as a user other than `root` with `sudo` permission):


```console
$ curl -L https://nixos.org/nix/install | sh
```

Make sure to follow the instructions output by the script.

The installation script requires that you have `sudo` access to `root`.

## Verify installation

You may want to verify the integrity of the installation script using GPG:

```console
$ curl -o install-nix-2.4 https://releases.nixos.org/nix/nix-2.4/install
$ curl -o install-nix-2.4.asc https://releases.nixos.org/nix/nix-2.4/install.asc
$ gpg2 --recv-keys B541D55301270E0BCF15CA5D8170B4726D7198DE
$ gpg2 --verify ./install-nix-2.4.asc
$ sh ./install-nix-2.4
```
The [signing key](https://nixos.org/edolstra.gpg) has fingerprint `B541 D553 0127 0E0B CF15 CA5D 8170 B472 6D71 98DE`. It is also available on [GitHub](https://github.com/NixOS/nixos-homepage/blob/master/edolstra.gpg).

## Uninstall Nix

You can uninstall Nix simply by running `rm -rf /nix`.

# Build with NixOS

If you use NixOS, nix-shell is already included.

## More on Nix

[Nix Manual](https://nixos.org/manual/nix/stable). Please read the [“Quick Start” section of the manual](https://nixos.org/manual/nix/stable/quick-start.html) for an overview of how to install and use Nix.

The `shell.nix` provides an environment containing the necessary dependencies. To enter the build environment, run (inside the build folder):

```console
$ nix-shell
```

Then run (from within the nix-shell environment):

```console
$ make
```

This will enter the environment and build the project. Note, that `nix-shell` provides an emulation of a common Linux
environment rather than the full-featured Nix package expression. No exportable Nix package will appear,
but local development is possible.

# Build with Nix on Windows

You could try running nix on [WSL](https://docs.microsoft.com/en-us/windows/wsl/). Not sure if it'll work.

Here are some resources that might help:

* [NixOS on WSL(2)](https://github.com/Trundle/NixOS-WSL)
* [Get started with the Windows Subsystem for Linux](https://docs.microsoft.com/en-us/learn/modules/get-started-with-windows-subsystem-for-linux/)