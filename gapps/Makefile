ifeq ($(USER), gharbi)
TORCH_INC ?= `python -c 'import torch.utils.ffi as ffi; print("-I"+" -I".join(ffi._setup_wrapper(True)[1]))'`
PYTHON=python
else
TORCH_INC ?= `python3 -c 'import torch.utils.ffi as ffi; print("-I"+" -I".join(ffi._setup_wrapper(True)[1]))'`
PYTHON=python3
endif
HALIDE_DIR ?=
AUTO_SCHEDULE = false
BUILD_TYPE ?= release

CXX ?= g++
CXXFLAGS += -std=c++11 -fno-rtti
INCLUDE=-I$(HALIDE_DIR)/include/
LDFLAGS ?=
LDFLAGS += $(HALIDE_DIR)/lib/libHalide.a -lpthread -ldl -lcurses -lz

# Cuda config
NVCC = nvcc -std c++11 -O2 #-G  -pg
NVFLAGS = -x cu -Xcompiler -fPIC -I$(SRC_DIR) \
					-gencode=arch=compute_30,code=\"sm_30,compute_30\" \
					-expt-relaxed-constexpr -Wno-deprecated-gpu-targets \
					-ftz=true --ptxas-options=-v -lineinfo
CUDA_LDFLAGS= -L/usr/local/cuda/lib64 -lcuda -lcudart

TARGET_FEATURES ?=

ifeq ($(BUILD_TYPE), profile)
TARGET_FEATURES = -profile
endif

ifeq ($(BUILD_TYPE), debug)
CXXFLAGS += -g -rdynamic
NVFLAGS += -g -G -lineinfo
else
CXXFLAGS += -O3
endif

ifeq ($(UNAME), Darwin)
CXXFLAGS += -fvisibility=hidden
endif

ifeq ($(UNAME), Darwin)
DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH):$(HALIDE_DIR)/bin
else
endif


HALIDE_GEN = $(HALIDE_DIR)/tools/GenGen.cpp

SRC_DIR = src
BUILD_DIR = build
EXT_DIR = _ext

# The makefile assumes that each op that has a backward also has a forward
OPS = \
			naive_demosaick_forward naive_demosaick_backward \
			bilateral_grid_forward bilateral_grid_backward \
			#bilateral_slice_apply_forward bilateral_slice_apply_backward \
			#bilateral_slice_forward bilateral_slice_backward \
			#bilateral_layer_forward bilateral_layer_backward \
			# conv2d_forward conv2d_general_scatter_forward \
			# bilinear_resampling_forward bilinear_resampling_backward \
			# spatial_transformer_forward spatial_transformer_backward \
			# deconv_grad_forward deconv_grad_backward \
			# deconv_alpha_forward deconv_alpha_backward \
			# deconv_prior_forward deconv_prior_backward
			# fancy_demosaick_forward fancy_demosaick_backward \
			# learnable_demosaick_forward learnable_demosaick_backward \
			# vgg_forward_backward \
			# burst_demosaicking_forward burst_demosaicking_backward \
			# deconv_cg_init_forward deconv_cg_init_backward deconv_cg_iter_forward deconv_cg_iter_backward \
			# deconv_cg_weight_forward deconv_cg_weight_backward \
			# deconv_grad_init_forward deconv_grad_init_backward \
			# deconv_grad_iter_forward deconv_grad_iter_backward \
			# non_local_means_forward non_local_means_backward \
			# deconv_cg_init_forward deconv_cg_init_backward deconv_cg_iter_forward deconv_cg_iter_backward \
			# deconv_cg_weight_forward deconv_cg_weight_backward bilateral_grid_forward bilateral_grid_backward \
			# conv1d_forward conv1d_backward conv1d_manual_backward \
			# histogram_forward histogram_backward \
			# soft_histogram_forward soft_histogram_backward

CUDA_SRC = bilateral_slice
CUDA_OBJ = $(addsuffix _cuda.so, $(addprefix $(BUILD_DIR)/, $(CUDA_SRC)))
OPS_LIBS = $(addsuffix .a, $(addprefix $(BUILD_DIR)/, $(OPS)))
CUDA_OPS_LIBS = $(addsuffix _cuda.a, $(addprefix $(BUILD_DIR)/, $(OPS)))

_ext/operators/_operators.so: $(OPS_LIBS) $(CUDA_OPS_LIBS) $(CUDA_OBJ)
	@echo Python CFFI wrapper
	@$(PYTHON) build.py

$(BUILD_DIR)/%_cuda.so: $(SRC_DIR)/cuda_kernels/%.cu.cc $(SRC_DIR)/cuda_kernels/%.h $(BUILD_DIR)
		$(NVCC) -c  $< -o $@ $(NVFLAGS) $(TORCH_INC)

$(BUILD_DIR)/%_forward.a: $(BUILD_DIR)/%_forward
	@echo Op $(subst $(BUILD_DIR)/,, $@)
	@cd $(BUILD_DIR); \
	DYLD_LIBRARY_PATH=$(HALIDE_DIR)/bin LD_LIBRARY_PATH=$(HALIDE_DIR)/bin FLUSH_MEMOIZE_CACHE=0 \
										./$(basename $(@F)) -g $(basename $(@F)) \
										-e static_library,h,pytorch_wrapper,html\
										-o . target=host$(TARGET_FEATURES) \
										auto_schedule=$(AUTO_SCHEDULE)

$(BUILD_DIR)/%_backward.a: $(BUILD_DIR)/%_backward
	@echo Op $(subst $(BUILD_DIR)/,, $@)
	@cd $(BUILD_DIR); \
	DYLD_LIBRARY_PATH=$(HALIDE_DIR)/bin LD_LIBRARY_PATH=$(HALIDE_DIR)/bin FLUSH_MEMOIZE_CACHE=1 \
										./$(basename $(@F)) -g $(basename $(@F)) \
										-e static_library,h,pytorch_wrapper,html\
										-o . target=host$(TARGET_FEATURES) \
										auto_schedule=$(AUTO_SCHEDULE)

$(BUILD_DIR)/%_forward_cuda.a: $(BUILD_DIR)/%_forward
	@$(eval name = $(subst _cuda,,$(basename $(@F))))
	@echo "Op (CUDA) $(name)"
	@cd $(BUILD_DIR); \
	DYLD_LIBRARY_PATH=$(HALIDE_DIR)/bin LD_LIBRARY_PATH=$(HALIDE_DIR)/bin FLUSH_MEMOIZE_CACHE=0 \
										./$(name) -g $(name) -f $(name)_cuda \
										-e static_library,pytorch_wrapper,h,html \
										-o . target=host-cuda-cuda_capability_61-user_context$(TARGET_FEATURES) \
										auto_schedule=$(AUTO_SCHEDULE)

$(BUILD_DIR)/%_backward_cuda.a: $(BUILD_DIR)/%_backward
	@$(eval name = $(subst _cuda,,$(basename $(@F))))
	@echo "Op (CUDA) $(name)"
	@cd $(BUILD_DIR); \
	DYLD_LIBRARY_PATH=$(HALIDE_DIR)/bin LD_LIBRARY_PATH=$(HALIDE_DIR)/bin FLUSH_MEMOIZE_CACHE=1 \
										./$(name) -g $(name) -f $(name)_cuda \
										-e static_library,pytorch_wrapper,h,html \
										-o . target=host-cuda-cuda_capability_61-user_context$(TARGET_FEATURES) \
										auto_schedule=$(AUTO_SCHEDULE)

# Generators, prevent auto-deletion by make
GENERATORS = $(addprefix $(BUILD_DIR)/, $(OPS))
.SECONDARY: $(GENERATORS)

$(BUILD_DIR)/%: $(SRC_DIR)/%.hl.cxx $(HALIDE_GEN) $(BUILD_DIR)
	@echo Generator $(subst $(BUILD_DIR)/,, $@)
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(HALIDE_GEN) $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -MMD -MP -o $(basename $@)

$(BUILD_DIR):
	@echo Making $@ dir
	@mkdir -p $@

clean:
	$(RM) -r $(BUILD_DIR) $(EXT_DIR)

DEPS = $(wildcard $(BUILD_DIR)/*.d)

-include $(DEPS)
