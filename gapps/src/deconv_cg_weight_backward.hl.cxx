#include "gradient_helpers.h"

#include "algorithms/deconv_cg_weight.h"

namespace gradient_apps {

class DeconvCgWeightBackwardGenerator
  : public Generator<DeconvCgWeightBackwardGenerator> {
public:
    Input<Buffer<float>>  blurred{"blurred", 3};
    Input<Buffer<float>>  current{"current", 3};
    Input<Buffer<float>>  reg_kernels{"reg_kernels", 3};
    Input<Buffer<float>>  reg_target_kernels{"reg_target_kernels", 3};
    Input<Buffer<float>>  reg_powers{"reg_powers", 1};
    Input<Buffer<float>>  d_weights{"d_weights", 4};
    Output<Buffer<float>> d_current{"d_current", 3};
    Output<Buffer<float>> d_reg_kernels{"d_reg_kernels", 3};
    Output<Buffer<float>> d_reg_target_kernels{"d_reg_target_kernels", 3};
    Output<Buffer<float>> d_reg_powers{"d_reg_powers", 1};

    void generate() {
        auto func_map = deconv_cg_weight(blurred, current,
            reg_kernels, reg_target_kernels, reg_powers);
        Func current_func = func_map["current_func"];
        Func reg_kernels_func = func_map["reg_kernels_func"];
        Func reg_target_kernels_func = func_map["reg_target_kernels_func"];
        Func reg_powers_func = func_map["reg_powers_func"];
        Func weights = func_map["weights"];
        Derivative d = propagate_adjoints(
            weights,
            d_weights,
            {{d_weights.dim(0).min(), d_weights.dim(0).max()},
             {d_weights.dim(1).min(), d_weights.dim(1).max()},
             {d_weights.dim(2).min(), d_weights.dim(2).max()},
             {d_weights.dim(3).min(), d_weights.dim(3).max()}}
        );
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assign_gradient(adjoints, current_func, d_current);
        assign_gradient(adjoints, reg_kernels_func, d_reg_kernels);
        assign_gradient(adjoints, reg_target_kernels_func, d_reg_target_kernels);
        assign_gradient(adjoints, reg_powers_func, d_reg_powers);

        if (auto_schedule) {
        } else {
            auto func_map = get_deps(
                {d_current, d_reg_kernels, d_reg_target_kernels, d_reg_powers}); 
            compute_all_root(d_current);
            compute_all_root(d_reg_kernels);
            compute_all_root(d_reg_target_kernels);
            compute_all_root(d_reg_powers);

            Func rKc = Func(func_map["rKc"]);
            rKc.update()
               .parallel(y)
               .vectorize(x, 16);
            Func rtKb = Func(func_map["rtKb"]);
            rtKb.update()
                .parallel(y)
                .vectorize(x, 16);

            Func d_repeat_edge_1 = Func(func_map["repeat_edge$1_0_d_def__"]);
            d_repeat_edge_1.update()
                           .parallel(y)
                           .vectorize(x, 16);

            Func d_rKc = Func(func_map["rKc_1_d_def__"]);
            d_rKc.update()
                 .parallel(y)
                 .vectorize(x, 16);

            Func d_rtKb = Func(func_map["rtKb_1_d_def__"]);
            d_rtKb.update()
                  .parallel(y)
                  .vectorize(x, 16);

            Var xy("xy"), xyn("xyn");
            RVar rxo("rxo"), rxi("rxi");
            Var rxi_f("rxi_f");
            Func d_rk = Func(func_map["reg_kernels_func_0_d_def__"]);
            auto d_rk_r0 = d_rk.rvars(0);

            d_rk.compute_root()
                .update(0)
                .split(d_rk_r0[0], rxo, rxi, 16);
            Func d_rk_rxo = d_rk.update()
                                .rfactor({{rxi, rxi_f}});

            d_rk.update(0)
                .fuse(x, y, xy)
                .fuse(xy, n, xyn)
                .parallel(xyn);

            d_rk_rxo.compute_at(d_rk, xyn)
                     .update()
                     .vectorize(rxi_f, 16);

            Func d_rtk = Func(func_map["reg_target_kernel_func_0_d_def__"]);
            auto d_rtk_r0 = d_rtk.rvars(0);

            d_rtk.compute_root()
                 .update(0)
                 .split(d_rtk_r0[0], rxo, rxi, 16);
            Func d_rtk_rxi = d_rtk.update()
                                  .rfactor({{rxi, rxi_f}});

            d_rtk.update(0)
                 .fuse(x, y, xy)
                 .fuse(xy, n, xyn)
                 .parallel(xyn);

            d_rtk_rxi.compute_at(d_rtk, xyn)
                     .update()
                     .vectorize(rxi_f, 16);

            Func d_reg_powers = Func(func_map["reg_powers_func_0_d_def__"]);
            auto d_reg_powers_r0 = d_reg_powers.rvars(0);

            Var ry("ry");
            Func d_reg_powers_ryo = d_reg_powers.update()
                                                .rfactor({{d_reg_powers_r0[1], ry}});

            d_reg_powers_ryo.compute_root()
                            .update()
                            .parallel(ry);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::DeconvCgWeightBackwardGenerator, deconv_cg_weight_backward)
