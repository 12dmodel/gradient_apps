#include "algorithms/bilateral_slice.h"
#include "gradient_helpers.h"

namespace gradient_apps {

class BilateralSliceForwardGenerator : public Generator<BilateralSliceForwardGenerator> {
public:
    Input<Buffer<float>>  grid{"grid", 5};   
    Input<Buffer<float>>  guide{"guide", 3};  
    Output<Buffer<float>> output{"output", 4};

    void generate() {
        std::map<std::string, Func> func_map = bilateral_slice(
            grid, guide);
        Func f_output = func_map["output"];
        output(x, y, co, n) = f_output(x, y, co, n);
        Func output_func = output;

        if(auto_schedule) {
        } else {
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            simple_autoschedule(output_func,
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
                                  {"guide.extent.2", 4}
                                },
                                {{0, 2047}, {0, 2047}, {0, 2}, {0, 3}},
                                options);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralSliceForwardGenerator, bilateral_slice_forward)
