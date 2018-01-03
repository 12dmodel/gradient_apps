#include "gradient_helpers.h"

#include "algorithms/deconv_cg_init.h"

namespace gradient_apps {

class DeconvCgInitForwardGenerator
  : public Generator<DeconvCgInitForwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  x0{"x0", 3};
    Input<Buffer<float>>  kernel{"kernel", 2};
    Input<Buffer<float>>  data_kernel_weights{"data_kernel_weights", 1};
    Input<Buffer<float>>  data_kernels{"data_kernels", 3};
    Input<Buffer<float>>  reg_kernel_weights{"reg_kernel_weights", 1};
    Input<Buffer<float>>  reg_kernels{"reg_kernels", 3};
    Input<Buffer<float>>  reg_targets{"reg_targets", 4};
    Input<Buffer<float>>  precond_kernel{"precond_kernel", 2};
    Input<Buffer<float>>  w_data{"w_data", 4};
    Input<Buffer<float>>  w_reg{"w_reg", 4};
    Output<Buffer<float>> xrp{"xrp", 4};

    void generate() {
        auto func_map = deconv_cg_init(blurred, x0, kernel,
            data_kernel_weights, data_kernels,
            reg_kernel_weights, reg_kernels, reg_targets,
            precond_kernel, w_data, w_reg);
        assert(func_map.find("xrp") != func_map.end());
        xrp(x, y, c, n) = func_map["xrp"](x, y, c, n);

        if (auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            Func xrp_func = xrp;
            simple_autoschedule(xrp_func,
                                {
                                 {"blurred.min.0", 0},
                                 {"blurred.min.1", 0},
                                 {"blurred.min.2", 0},
                                 {"blurred.extent.0", 256},
                                 {"blurred.extent.1", 256},
                                 {"blurred.extent.2", 3},
                                 {"x0.min.0", 0},
                                 {"x0.min.1", 0},
                                 {"x0.min.2", 0},
                                 {"x0.extent.0", 256},
                                 {"x0.extent.1", 256},
                                 {"x0.extent.2", 3},
                                 {"kernel.min.0", 0},
                                 {"kernel.min.1", 0},
                                 {"kernel.extent.0", 11},
                                 {"kernel.extent.1", 11},
                                 {"data_kernel_weights.min.0", 0},
                                 {"data_kernel_weights.extent.0", 5},
                                 {"data_kernels.min.0", 0},
                                 {"data_kernels.min.1", 0},
                                 {"data_kernels.min.2", 0},
                                 {"data_kernels.extent.0", 5},
                                 {"data_kernels.extent.1", 5},
                                 {"data_kernels.extent.2", 5},
                                 {"reg_kernel_weights.min.0", 0},
                                 {"reg_kernel_weights.extent.0", 5},
                                 {"reg_kernels.min.0", 0},
                                 {"reg_kernels.min.1", 0},
                                 {"reg_kernels.min.2", 0},
                                 {"reg_kernels.extent.0", 5},
                                 {"reg_kernels.extent.1", 5},
                                 {"reg_kernels.extent.2", 5},
                                 {"reg_targets.min.0", 0},
                                 {"reg_targets.min.1", 0},
                                 {"reg_targets.min.2", 0},
                                 {"reg_targets.min.3", 0},
                                 {"reg_targets.extent.0", 256},
                                 {"reg_targets.extent.1", 256},
                                 {"reg_targets.extent.2", 3},
                                 {"reg_targets.extent.3", 5},
                                 {"precond_kernel.min.0", 0},
                                 {"precond_kernel.min.1", 0},
                                 {"precond_kernel.extent.0", 11},
                                 {"precond_kernel.extent.1", 11},
                                 {"w_data.min.0", 0},
                                 {"w_data.min.1", 0},
                                 {"w_data.min.2", 0},
                                 {"w_data.min.3", 0},
                                 {"w_data.extent.0", 256},
                                 {"w_data.extent.1", 256},
                                 {"w_data.extent.2", 3},
                                 {"w_data.extent.3", 5},
                                 {"w_reg.min.0", 0},
                                 {"w_reg.min.1", 0},
                                 {"w_reg.min.2", 0},
                                 {"w_reg.min.3", 0},
                                 {"w_reg.extent.0", 256},
                                 {"w_reg.extent.1", 256},
                                 {"w_reg.extent.2", 3},
                                 {"w_reg.extent.3", 5}
                                },
                                {{0, 255},
                                 {0, 255},
                                 {0, 2},
                                 {0, 2}},
                                options,
                                {"xrp$1"});
#if 0
            auto func_map = get_deps(xrp);
            compute_all_root(xrp);
            Func Kx0 = Func(func_map["Kx0"]);
            Kx0.update()
               .parallel(y)
               .vectorize(x, 16);
            Func KTWKx0 = Func(func_map["K^TWKx0"]);
            KTWKx0.update()
                  .parallel(y)
                  .vectorize(x, 16);
            Func rKx0 = Func(func_map["rKx0"]);
            rKx0.update()
                .parallel(y)
                .vectorize(x, 16);
            Func rKTWrKx0 = Func(func_map["rK^TWrKx0"]);
            rKTWrKx0.update()
                    .parallel(y)
                    .vectorize(x, 16);
            Func KTWb = Func(func_map["K^TWb"]);
            KTWb.update()
                .parallel(y)
                .vectorize(x, 16);
            Func Pr0 = Func(func_map["Pr0"]);
            Pr0.update()
               .parallel(y)
               .vectorize(x, 16);
            Func z0 = Func(func_map["z0$0"]);
            z0.update()
              .parallel(y)
              .vectorize(x, 16);
#endif
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgInitForwardGenerator, deconv_cg_init_forward)
