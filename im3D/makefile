# Directory names:
BUILD = ./build
INSTALL = ../bin

HIST_SRC = histogram/pyx_src
METRICS_SRC = metrics/pyx_src
TRANSFORM_SRC = transform/pyx_src
SMOOTHING_SRC = smoothing/pyx_src

# Python, Cython and C compiler executables:
PY = /opt/local/bin/py27-MacPorts
CY = /opt/local/bin/cython-2.7
CC = /usr/local/bin/gcc

# Compiling:
CC_OPT=-O3 -fPIC -fopenmp -march=native
INC_DIRS=-I/opt/local/Library/Frameworks/python.framework/Versions/2.7/include/python2.7

# Linking:
LINK_DIRS=-L/opt/local/Library/Frameworks/python.framework/Versions/2.7/lib
LINK_LIBS=-lpython2.7
LINK_OPTS=-shared -fopenmp

# Shared object files:
HIST_SO = histogram_1D.so histogram_2D.so
METRICS_SO = metrics_rms.so metrics_sv.so metrics_delta.so
TRANSFORM_SO = translate.so translate_2D.so translate_3D.so \
               rotate.so rotate_2D.so rotate_3D.so
SMOOTHING_SO = DS_1D.so  DS_2D.so  DS_3D.so DS_4D.so  \
               MMC_2D.so  MMC_3D.so
SO_FILES = $(HIST_SO) $(METRICS_SO) $(TRANSFORM_SO) $(SMOOTHING_SO)


############## Default make case ###############
all: $(SO_FILES)


################################################################
############            HISTOGRAM MODULE            ############
################################################################
histogram: $(HIST_SO)


### Cython:
$(BUILD)/histogram_1D.c: $(HIST_SRC)/histogram_1D.pyx | $(BUILD)
	$(CY) -a $(HIST_SRC)/histogram_1D.pyx -o $(BUILD)/histogram_1D.c
### Compile:
$(BUILD)/histogram_1D.o: $(BUILD)/histogram_1D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/histogram_1D.c -o $(BUILD)/histogram_1D.o
### Link:
histogram_1D.so: $(BUILD)/histogram_1D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/histogram_1D.o -o histogram_1D.so


### Cython:
$(BUILD)/histogram_2D.c: $(HIST_SRC)/histogram_2D.pyx | $(BUILD)
	$(CY) -a $(HIST_SRC)/histogram_2D.pyx -o $(BUILD)/histogram_2D.c
### Compile:
$(BUILD)/histogram_2D.o: $(BUILD)/histogram_2D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/histogram_2D.c -o $(BUILD)/histogram_2D.o
### Link:
histogram_2D.so: $(BUILD)/histogram_2D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/histogram_2D.o -o histogram_2D.so


################################################################
############             METRICS MODULE             ############
################################################################
metrics: $(METRICS_SO)


### Cython:
$(BUILD)/metrics_sv.c: $(METRICS_SRC)/metrics_sv.pyx | $(BUILD)
	$(CY) -a $(METRICS_SRC)/metrics_sv.pyx -o $(BUILD)/metrics_sv.c
### Compile:
$(BUILD)/metrics_sv.o: $(BUILD)/metrics_sv.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/metrics_sv.c -o $(BUILD)/metrics_sv.o
### Link:
metrics_sv.so: $(BUILD)/metrics_sv.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/metrics_sv.o -o metrics_sv.so


### Cython:
$(BUILD)/metrics_rms.c: $(METRICS_SRC)/metrics_rms.pyx | $(BUILD)
	$(CY) -a $(METRICS_SRC)/metrics_rms.pyx -o $(BUILD)/metrics_rms.c
### Compile:
$(BUILD)/metrics_rms.o: $(BUILD)/metrics_rms.c $(METRICS_SRC)/metrics_rms.pyx
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/metrics_rms.c -o $(BUILD)/metrics_rms.o
### Link:
metrics_rms.so: $(BUILD)/metrics_rms.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/metrics_rms.o -o metrics_rms.so


### Cython:
$(BUILD)/metrics_delta.c: $(METRICS_SRC)/metrics_delta.pyx | $(BUILD)
	$(CY) -a $(METRICS_SRC)/metrics_delta.pyx -o $(BUILD)/metrics_delta.c
### Compile:
$(BUILD)/metrics_delta.o: $(BUILD)/metrics_delta.c $(METRICS_SRC)/metrics_delta.pyx
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/metrics_delta.c -o $(BUILD)/metrics_delta.o
### Link:
metrics_delta.so: $(BUILD)/metrics_delta.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/metrics_delta.o -o metrics_delta.so


################################################################
############            TRANSFORM MODULE            ############
################################################################
metrics: $(TRANSFORM_SO)


### Cython:
$(BUILD)/translate.c: $(TRANSFORM_SRC)/translate.pyx | $(BUILD)
	$(CY) -a $(TRANSFORM_SRC)/translate.pyx -o $(BUILD)/translate.c
### Compile:
$(BUILD)/translate.o: $(BUILD)/translate.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/translate.c -o $(BUILD)/translate.o
### Link:
translate.so: $(BUILD)/translate.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/translate.o -o translate.so


### Cython:
$(BUILD)/translate_2D.c: $(TRANSFORM_SRC)/translate_2D.pyx | $(BUILD)
	$(CY) -a $(TRANSFORM_SRC)/translate_2D.pyx -o $(BUILD)/translate_2D.c
### Compile:
$(BUILD)/translate_2D.o: $(BUILD)/translate_2D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/translate_2D.c -o $(BUILD)/translate_2D.o
### Link:
translate_2D.so: $(BUILD)/translate_2D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/translate_2D.o -o translate_2D.so


### Cython:
$(BUILD)/translate_3D.c: $(TRANSFORM_SRC)/translate_3D.pyx | $(BUILD)
	$(CY) -a $(TRANSFORM_SRC)/translate_3D.pyx -o $(BUILD)/translate_3D.c
### Compile:
$(BUILD)/translate_3D.o: $(BUILD)/translate_3D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/translate_3D.c -o $(BUILD)/translate_3D.o
### Link:
translate_3D.so: $(BUILD)/translate_3D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/translate_3D.o -o translate_3D.so


### Cython:
$(BUILD)/rotate.c: $(TRANSFORM_SRC)/rotate.pyx | $(BUILD)
	$(CY) -a $(TRANSFORM_SRC)/rotate.pyx -o $(BUILD)/rotate.c
### Compile:
$(BUILD)/rotate.o: $(BUILD)/rotate.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/rotate.c -o $(BUILD)/rotate.o
### Link:
rotate.so: $(BUILD)/rotate.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/rotate.o -o rotate.so


### Cython:
$(BUILD)/rotate_2D.c: $(TRANSFORM_SRC)/rotate_2D.pyx | $(BUILD)
	$(CY) -a $(TRANSFORM_SRC)/rotate_2D.pyx -o $(BUILD)/rotate_2D.c
### Compile:
$(BUILD)/rotate_2D.o: $(BUILD)/rotate_2D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/rotate_2D.c -o $(BUILD)/rotate_2D.o
### Link:
rotate_2D.so: $(BUILD)/rotate_2D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/rotate_2D.o -o rotate_2D.so


### Cython:
$(BUILD)/rotate_3D.c: $(TRANSFORM_SRC)/rotate_3D.pyx | $(BUILD)
	$(CY) -a $(TRANSFORM_SRC)/rotate_3D.pyx -o $(BUILD)/rotate_3D.c
### Compile:
$(BUILD)/rotate_3D.o: $(BUILD)/rotate_3D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/rotate_3D.c -o $(BUILD)/rotate_3D.o
### Link:
rotate_3D.so: $(BUILD)/rotate_3D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/rotate_3D.o -o rotate_3D.so


################################################################
############            SMOOTHING MODULE            ############
################################################################
smoothing: $(SMOOTHING_SO)


#################### DS_1D ####################
### Cython:
$(BUILD)/DS_1D.c: $(SMOOTHING_SRC)/DS_1D.pyx | $(BUILD)
	$(CY) -a $(SMOOTHING_SRC)/DS_1D.pyx -o $(BUILD)/DS_1D.c
### Compile:
$(BUILD)/DS_1D.o: $(BUILD)/DS_1D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/DS_1D.c -o $(BUILD)/DS_1D.o
### Link:
DS_1D.so: $(BUILD)/DS_1D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/DS_1D.o -o DS_1D.so


#################### DS_2D ####################
### Cython:
$(BUILD)/DS_2D.c: $(SMOOTHING_SRC)/DS_2D.pyx | $(BUILD)
	$(CY) -a $(SMOOTHING_SRC)/DS_2D.pyx -o $(BUILD)/DS_2D.c
### Compile:
$(BUILD)/DS_2D.o: $(BUILD)/DS_2D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/DS_2D.c -o $(BUILD)/DS_2D.o
### Link:
DS_2D.so: $(BUILD)/DS_2D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/DS_2D.o -o DS_2D.so


#################### DS_3D ####################
### Cython:
$(BUILD)/DS_3D.c: $(SMOOTHING_SRC)/DS_3D.pyx | $(BUILD)
	$(CY) -a $(SMOOTHING_SRC)/DS_3D.pyx -o $(BUILD)/DS_3D.c
### Compile:
$(BUILD)/DS_3D.o: $(BUILD)/DS_3D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/DS_3D.c -o $(BUILD)/DS_3D.o
### Link:
DS_3D.so: $(BUILD)/DS_3D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/DS_3D.o -o DS_3D.so


#################### DS_4D ####################
### Cython:
$(BUILD)/DS_4D.c: $(SMOOTHING_SRC)/DS_4D.pyx | $(BUILD)
	$(CY) -a $(SMOOTHING_SRC)/DS_4D.pyx -o $(BUILD)/DS_4D.c
### Compile:
$(BUILD)/DS_4D.o: $(BUILD)/DS_4D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/DS_4D.c -o $(BUILD)/DS_4D.o
### Link:
DS_4D.so: $(BUILD)/DS_4D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/DS_4D.o -o DS_4D.so


#################### MMC_2D ####################
### Cython:
$(BUILD)/MMC_2D.c: $(SMOOTHING_SRC)/MMC_2D.pyx | $(BUILD)
	$(CY) -a $(SMOOTHING_SRC)/MMC_2D.pyx -o $(BUILD)/MMC_2D.c
### Compile:
$(BUILD)/MMC_2D.o: $(BUILD)/MMC_2D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/MMC_2D.c -o $(BUILD)/MMC_2D.o
### Link:
MMC_2D.so: $(BUILD)/MMC_2D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/MMC_2D.o -o MMC_2D.so


#################### MMC_3D ####################
### Cython:
$(BUILD)/MMC_3D.c: $(SMOOTHING_SRC)/MMC_3D.pyx | $(BUILD)
	$(CY) -a $(SMOOTHING_SRC)/MMC_3D.pyx -o $(BUILD)/MMC_3D.c
### Compile:
$(BUILD)/MMC_3D.o: $(BUILD)/MMC_3D.c
	$(CC) $(INC_DIRS) $(CC_OPT) -c $(BUILD)/MMC_3D.c -o $(BUILD)/MMC_3D.o
### Link:
MMC_3D.so: $(BUILD)/MMC_3D.o
	$(CC) $(LINK_DIRS) $(LINK_LIBS) $(LINK_OPTS) build/MMC_3D.o -o MMC_3D.so



################################################################
############             MISC UTILITIES             ############
################################################################
# Build dir:
$(BUILD):
	mkdir -p $(BUILD)


install: $(HIST_SO) $(METRICS_SO) $(TRANSFORM_SO) $(SMOOTHING_SO) | $(INSTALL)
	mv $(HIST_SO) $(INSTALL)
	mv $(METRICS_SO) $(INSTALL)
	mv $(TRANSFORM_SO) $(INSTALL)
	mv $(SMOOTHING_SO) $(INSTALL)
#	for FILENAME in $(SO_FILES); do \
#		mv $$FILENAME $(INSTALL) ; \
#	done

clean:
	rm -rf *.so
	rm -rf *.pyc
	rm -rf build

