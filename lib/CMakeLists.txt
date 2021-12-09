add_library(
	c-skl
	src/c-skl/skl_helper.h
	src/c-skl/skl_helper.c
	src/c-skl/linear_model/linear_regression/skl_linear_regression.h
	src/c-skl/linear_model/linear_regression/skl_linear_regression.c
	src/c-skl/svm/svr/skl_svr.h
	src/c-skl/svm/svr/skl_svr.c
)

target_include_directories(c-skl PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
