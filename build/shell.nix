{ pkgs ? import <nixpkgs> {} }:
let
  py3 = pkgs.python3;
in pkgs.mkShell {
  buildInputs = [
    py3.pkgs.numpy
    py3.pkgs.scikit-learn
  ];
  NIX_CFLAGS_COMPILE = [
    "-isystem ${py3}/include/${py3.libPrefix}"
    "-isystem ${py3.pkgs.numpy}/${py3.sitePackages}/numpy/core/include"
  ];
  NIX_LDFLAGS = [
    "-l${py3.libPrefix}"
     ];
  shellHook = ''
  	printf "Configuring build environment...\n\n"
  	echo "If not done already, run cmake .."
  	echo "Then, run 'cmake --build .' to build the library"
  '';
}
