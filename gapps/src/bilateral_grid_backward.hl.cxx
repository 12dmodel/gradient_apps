#include "algorithms/bilateral_grid.h"
#include "gradient_helpers.h"

namespace gradient_apps {

class BilateralGridBackwardGenerator : public Generator<BilateralGridBackwardGenerator> {
public:
    Input<Buffer<float>>  input{"input", 4};       // x, y, channel, batch
    Input<Buffer<float>>  guide{"guide", 3};       // x, y, batch
    Input<Buffer<float>>  filter_s{"filter_s", 1};
    Input<Buffer<float>>  filter_r{"filter_r", 1};
    Input<Buffer<float>>  d_output{"d_output", 4};

    Output<Buffer<float>> d_input{"d_input", 4};
    Output<Buffer<float>> d_guide{"d_guide", 3};
    Output<Buffer<float>> d_filter_s{"d_filter_s", 1};
    Output<Buffer<float>> d_filter_r{"d_filter_r", 1};

    void generate() {
        std::map<std::string, Func> func_map = bilateral_grid(
            input, guide, filter_s, filter_r);
        Func f_output = func_map["output"];
        Func f_input = func_map["f_input"];
        Func f_guide = func_map["f_guide"];
        Func f_filter_s = func_map["f_filter_s"];
        Func f_filter_r = func_map["f_filter_r"];
        Derivative d = propagate_adjoints(
            f_output,
            d_output,
            {{d_output.dim(0).min(), d_output.dim(0).max()},
             {d_output.dim(1).min(), d_output.dim(1).max()},
             {d_output.dim(2).min(), d_output.dim(2).max()},
             {d_output.dim(3).min(), d_output.dim(3).max()}}
        );
        std::map<FuncKey, Func> adjoints = d.adjoints;
        assign_gradient(adjoints, f_input, d_input);
        assign_gradient(adjoints, f_guide, d_guide);
        assign_gradient(adjoints, f_filter_s, d_filter_s);
        assign_gradient(adjoints, f_filter_r, d_filter_r);

        if(auto_schedule) {
        } else {
            std::vector<Func> funcs{d_input, d_guide, d_filter_s, d_filter_r};
            SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            options.gpu_tile_channel = 3;
            simple_autoschedule(funcs,
                                {{"input.min.0", 0},
                                 {"input.min.1", 0},
                                 {"input.min.2", 0},
                                 {"input.min.3", 0},
                                 {"input.extent.0", 256},
                                 {"input.extent.1", 256},
                                 {"input.extent.2", 3},
                                 {"input.extent.3", 4},
                                 {"guide.min.0", 0},
                                 {"guide.min.1", 0},
                                 {"guide.min.2", 0},
                                 {"guide.extent.0", 256},
                                 {"guide.extent.1", 256},
                                 {"guide.extent.2", 4},
                                 {"filter_s.min.0", 0},
                                 {"filter_s.extent.0", 5},
                                 {"filter_r.min.0", 0},
                                 {"filter_r.extent.0", 5},
                                 {"d_output.min.0", 0},
                                 {"d_output.min.1", 0},
                                 {"d_output.min.2", 0},
                                 {"d_output.min.3", 0},
                                 {"d_output.extent.0", 256},
                                 {"d_output.extent.1", 256},
                                 {"d_output.extent.2", 3},
                                 {"d_output.extent.3", 4}},
                                {{{0, 255},
                                  {0, 255},
                                  {0, 2},
                                  {0, 3}},
                                 {{0, 255},
                                  {0, 255},
                                  {0, 3}},
                                 {{0, 4}},
                                 {{0, 4}}},
                                options);
        }
    }
};

}  // end namespace gradient_apps

HALIDE_REGISTER_GENERATOR(
    gradient_apps::BilateralGridBackwardGenerator, bilateral_grid_backward)
