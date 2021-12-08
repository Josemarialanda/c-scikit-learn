OUT=out
CC=gcc

SOURCES_0 = $(wildcard *.c)
SOURCES_1 = $(wildcard skl/*.c)
SOURCES_2 = $(wildcard skl/linear_model/linear_regression/*.c)
SOURCES_3 = $(wildcard skl/svm/svr/*.c)

SOURCES = $(SOURCES_0) $(SOURCES_1) $(SOURCES_2) $(SOURCES_3)

all:
	$(CC) -Wall -g $(SOURCES) -o $(OUT)
