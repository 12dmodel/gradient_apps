#include "algorithms/bilateral_slice.h"
#include "gradient_helpers.h"

namespace gradient_apps {

class BilateralSliceBackwardGenerator : public Generator<BilateralSliceBackwardGenerator> {
public:
    Input<Buffer<float>>  grid{"grid", 5};   
    Input<Buffer<float>>  guide{"guide", 3};  
    Input<Buffer<float>> d_output{"d_output", 4};

    Output<Buffer<float>> d_grid{"d_grid", 5};
    Output<Buffer<float>> d_guide{"d_guide", 3};

    void generate() {
        std::map<std::string, Func> func_map = bilateral_slice(
            grid, guide);
        Func f_output = func_map["output"];
        Func f_grid = func_map["f_grid"];
        Func f_guide = func_map["f_guide"];

        Derivative d = propagate_adjoints(
            f_output, 
            d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()},
             {d_output.dim(3).min(), d_output.dim(3).max()}}
             );

        assign_gradient(d, f_grid, d_grid);
        assign_gradient(d, f_guide, d_guide);

        if(auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            std::vector<Func> funcs{d_grid, d_guide};
            simple_autoschedule(funcs,
                                {
                                  {"grid.min.0", 0},
                                  {"grid.min.1", 0},
                                  {"grid.min.2", 0},
                                  {"grid.min.3", 0},
                                  {"grid.min.4", 0},
                                  {"grid.extent.0", 64},
                                  {"grid.extent.1", 64},
                                  {"grid.extent.2", 8},
                                  {"grid.extent.3", 12},
                                  {"grid.extent.4", 4},
                                  {"guide.min.0", 0},
                                  {"guide.min.1", 0},
                                  {"guide.min.2", 0},
                                  {"guide.extent.0", 2048},
                                  {"guide.extent.1", 2048},
                                  {"guide.extent.2", 4},
                                  {"d_output.min.0", 0},
                                  {"d_output.min.1", 0},
                                  {"d_output.min.2", 0},
                                  {"d_output.min.3", 0},
                                  {"d_output.extent.0", 2048},
                                  {"d_output.extent.1", 2048},
                                  {"d_output.extent.2", 3},
                                  {"d_output.extent.3", 4}
                                },
                                {
                                  {{0, 63}, {0, 63}, {0, 7}, {0, 11}, {0, 3}},
                                  {{0, 2047}, {0, 2047}, {0, 3}}
                                },
                                options);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralSliceBackwardGenerator, bilateral_slice_backward)
