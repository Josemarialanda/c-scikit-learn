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
    array* x = skl_get_array(r,c);

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

# Install Instructions

Note: building this repository with `nix` requires version 2.4 or newer and flake support. Check your nix version with `nix --version` in a terminal.

# Build with Nix

## Install Nix

[NixOS - Getting Nix / NixOS](https://nixos.org/download.html#nix-install-linux)

## Enable flakes

**Nix Flakes** are an upcoming feature of the Nix package manager. We use flakes to pin **nixpkgs** to a specific version to ensure *hermetic* builds.

### Installing flakes

#### NixOS

In NixOS this can be achieved with the following options in `configuration.nix`.

**System-wide installation:**

```nix
{ pkgs, ... }: {
  nix = {
    package = pkgs.nixFlakes; # or versioned attributes like nix_2_7
    extraOptions = ''
      experimental-features = nix-command flakes
    '';
   };
}
```

#### Non-NixOS

On non-nixos systems, install `nixFlakes` in your environment:

```bash
$ nix-env -iA nixpkgs.nixFlakes
```

Edit either `~/.config/nix/nix.conf` or `/etc/nix/nix.conf` and add:

```bash
experimental-features = nix-command flakes
```

This is needed to expose the Nix 2.0 CLI and flakes support that are hidden behind feature-flags. Finally, if the Nix installation is in multi-user mode, don’t forget to restart the nix-daemon. There is no official installer yet, but you can use the [nix-unstable-installer](https://github.com/numtide/nix-unstable-installer#systems):

# Build with NixOS

If you use NixOS, nix is already included. One needs only enable flake support in `configuration.nix`. 

## More on Nix

[Nix Manual](https://nixos.org/manual/nix/stable). Please read the [“Quick Start” section of the manual](https://nixos.org/manual/nix/stable/quick-start.html) for an overview of how to install and use Nix.

The `shell.nix` provides an environment containing the necessary dependencies. To enter the build environment, run (inside the build folder):

```console
$ nix develop
```

Then run (from within the nix-shell environment):

```console
$ make
```

This will enter the environment and build the project. Note, that `nix-shell` provides an emulation of a common Linux environment rather than the full-featured Nix package expression. No exportable Nix package will appear, but local development is possible.