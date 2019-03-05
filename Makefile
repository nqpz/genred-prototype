NVCC=nvcc
# if -arch or -code is not specified, the default (CUDA 8.0) is
# compute_20 and (minimum supported cc).
# -arch=compute_35 -code=sm_35 or just -arch=sm_35
#NVCC_FLAGS:= -arch=compute_20 -code=sm_35 -x cu -Wno-deprecated-gpu-targets
NVCC_FLAGS := -x cu -Wno-deprecated-gpu-targets
C_OPTIONS := -Wall -Wextra -Werror -O3
C_FLAGS := $(foreach option, $(C_OPTIONS), --compiler-options $(option))
LIBS := -lm

REQS=kernels.cu.h misc.cu.h
CU_FILE=host

FUT_FILE=reduce
FB=opencl
FC=futhark $(FB)

# For experiment
DATA_PATH_CUDA=data/cuda
DATA_PATH_FUT=data/futhark
RUNT_PATH=runtimes
PDF_PATH=pdf
DATA_SIZE=10000000
ITERATIONS=5
COOP_LEVELS=1 4 16 32 64 256 1024 4096 16384 61440 # last is max threads
HISTO_SIZES=16 32 64 256 1024 4096 16384 61440

.PHONY: all plot run dat clean clean_runtimes clean_data clean_pfds clean_bins

.PRECIOUS: $(RUNT_PATH)/hist-%.json hist-%-full.json $(DATA_PATH_CUDA)/%-$(DATA_SIZE).dat $(DATA_PATH_CUDA)/futhark/%

all: $(CU_FILE)

# Compile CUDA prototype
$(CU_FILE): $(CU_FILE).cu $(REQS)
	$(NVCC) $(NVCC_FLAGS) $(C_FLAGS) $< -o $@ $(LIBS)

# Compile Futhark reduction program
$(FUT_FILE): $(FUT_FILE).fut
	$(FC) $<

# Run experiment
plot: $(HISTO_SIZES:%=$(PDF_PATH)/hist-%.pdf) $(HISTO_SIZES:%=$(PDF_PATH)/hist-%-full.pdf)

# Support performing runs on its own.
run: $(HISTO_SIZES:%=$(RUNT_PATH)/hist-%.json $(RUNT_PATH)/fut_times.json)

# Support generating data on its own.
dat: datcuda datfut
datcuda: $(DATA_PATH_CUDA) $(HISTO_SIZES:%=$(DATA_PATH_CUDA)/%-$(DATA_SIZE).dat)
datfut: $(DATA_PATH_FUT)

$(RUNT_PATH) $(PDF_PATH):
	mkdir -p $@

$(DATA_PATH_CUDA):
	mkdir -p $(DATA_PATH_CUDA)
	./generate_images_adverserial.py

$(DATA_PATH_FUT):
	mkdir -p $(DATA_PATH_FUT)
	./generate_fut_dat.sh

# Generate CUDA data (Futhark data should be created manually!)
$(DATA_PATH_CUDA)/%-$(DATA_SIZE).dat:
	@echo '=== Generating data'
	python generate_image.py $* $(DATA_SIZE)

# Run actual programs
$(RUNT_PATH)/hist-%.json: $(RUNT_PATH) $(CU_FILE) $(DATA_PATH_CUDA)/%-$(DATA_SIZE).dat
	@echo '=== Running CUDA experiment'
	python experiment.py $(ITERATIONS) $* \
	  $(DATA_PATH_CUDA)/$*-$(DATA_SIZE).dat $(COOP_LEVELS)

$(RUNT_PATH)/fut_times.json: $(DATA_PATH_FUT) $(RUNT_PATH) $(FUT_FILE).fut
	@echo '=== Running Futhark experiment'
	futhark bench --runs=$(ITERATIONS) --backend=$(FB) --json $@ $(FUT_FILE).fut

# Create graphs
$(PDF_PATH)/hist-%.pdf $(PDF_PATH)/hist-%-full.pdf: $(PDF_PATH) $(RUNT_PATH)/hist-%.json $(RUNT_PATH)/fut_times.json
	@echo '=== Generating graphs'
	python plot.py $* $(DATA_SIZE) $(COOP_LEVELS)

clean_runtimes:
	rm -fr $(RUNT_PATH)

clean_data:
#rm -r $(DATA_PATH_CUDA) $(DATA_PATH_FUT) # you probably don't want to do this

clean_pdfs:
	rm -fr $(PDF_PATH)

clean_bins:
	rm -f $(CU_FILE)
	rm -f $(FUT_FILE) $(FUT_FILE).c

clean: clean_runtimes clean_data clean_pdfs clean_bins
