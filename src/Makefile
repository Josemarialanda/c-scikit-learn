OUT=out
CC=gcc

SOURCES_0 = $(wildcard *.c)
SOURCES_1 = $(wildcard dataset_utils/*.c)
SOURCES_2 = $(wildcard skl/svm/linear_svc/*.c)
SOURCES_3 = $(wildcard skl/svm/linear_svr/*.c)
SOURCES_4 = $(wildcard skl/svm/nu_svc/*.c)
SOURCES_5 = $(wildcard skl/*.c)
SOURCES_6 = $(wildcard skl/linear_model/linear_regression/*.c)

SOURCES = $(SOURCES_0) $(SOURCES_1) $(SOURCES_2) $(SOURCES_3) $(SOURCES_4) $(SOURCES_5) $(SOURCES_6) 

all:
	$(CC) -Wall -g $(SOURCES) -o $(OUT)
