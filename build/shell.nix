{ pkgs ? import <nixpkgs> {} }:
let
  py3 = pkgs.python3;
  buildTools = with pkgs; [gnumake cmake ncurses];
in pkgs.mkShell {
  buildInputs = [
    py3.pkgs.numpy
    py3.pkgs.scikit-learn
  ] ++ buildTools;
  NIX_CFLAGS_COMPILE = [
    "-isystem ${py3}/include/${py3.libPrefix}"
    "-isystem ${py3.pkgs.numpy}/${py3.sitePackages}/numpy/core/include"
  ];
  NIX_LDFLAGS = [
    "-l${py3.libPrefix}"
     ];
  shellHook = ''  	
	FILE=Makefile
	
	alias run-example="./test-exe/main-exe"
	alias set-build-shared="cmake -D BUILD_SHARED_LIBS=TRUE ."
	alias set-build-static="cmake -D BUILD_SHARED_LIBS=FALSE ."
	
	if [ -f "$FILE" ]; then
		printf "$FILE found.\n"
		printf "To build the project, run 'make' from within the build folder."
	else 
		printf "$FILE not found. Configuring project.\n\n"
		cmake ..
		printf "\nTo build the project, run 'make' from within the build folder."
	fi
	  	
  '';
}
