SRC_DIR    ?= src
COMMON_DIR ?= ../common
EXEC        = miner

ifndef COMPILER
$(error COMPILER is not defined)
endif

PARTITION ?= ucsx
TIME      ?= 00:01:00
OBJS      := $(SRC_DIR)/miner.o $(SRC_DIR)/sha256.o $(SRC_DIR)/utils.o
LIBS      := -lm

ifeq ($(COMPILER), nvcc)
GRES    := --gres gpu:1
MODULE  := module load libraries/cuda-13.1 &&
FORCE_C :=
else
GRES    :=
MODULE  :=
FORCE_C := -x c
endif

build:
	@sbatch $(GRES) --time $(TIME) --partition $(PARTITION) --wrap="$(MODULE) make compile"

run:
	@sbatch $(GRES) --time $(TIME) --partition $(PARTITION) --wrap="$(MODULE) ./$(EXEC) $(TEST)"

compile: $(EXEC)

$(EXEC): $(OBJS)
	$(COMPILER) $(OBJS) -o $@ $(LIBS)

$(SRC_DIR)/miner.o: $(COMMON_DIR)/miner.cpp
	$(COMPILER) $(CFLAGS) $(FORCE_C) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.$(SRC_EXT)
	$(COMPILER) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o output/* $(EXEC) slurm-*.out slurm-*.err profile.ncu-rep

.PHONY: build run compile clean