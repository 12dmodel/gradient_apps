#define NEED_HESS
#include "deconv_grad_forward.h"

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvGradForwardGenerator, deconv_grad_hess_forward)
