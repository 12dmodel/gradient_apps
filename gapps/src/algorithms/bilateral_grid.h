#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), c("c"), n("n");

template <typename Input>
std::map<std::string, Func> bilateral_grid(
        const Input &input,
        const Input &guide,
        const Input &filter_s,
        const Input &filter_r) {
    int sigma_s = 2;
    int sigma_r = 32;

    Func f_input("f_input");
    f_input(x, y, c, n) =
      Halide::BoundaryConditions::repeat_edge(input)(x, y, c, n);
    Func f_guide("f_guide");
    f_guide(x, y, n) =
      Halide::BoundaryConditions::repeat_edge(guide)(x, y, n);
    Func f_filter_s("f_filter_s");
    f_filter_s(x) = filter_s(x);
    Func f_filter_r("f_filter_r");
    f_filter_r(x) = filter_r(x);

    // Downsample grid
    RDom rgrid(0, sigma_s, 0, sigma_s);
    Expr guide_pos = clamp(f_guide(x * sigma_s + rgrid.x,
                                 y * sigma_s + rgrid.y, n) * cast<float>(sigma_r),
                           0.f,
                           cast<float>(sigma_r));
    Expr lower_bin = cast<int>(floor(guide_pos));
    Expr upper_bin = cast<int>(ceil(guide_pos));
    Expr w = guide_pos - lower_bin;

    Expr val = select(c < input.channels(),
                      f_input(cast<int>(x * sigma_s + rgrid.x),
                              cast<int>(y * sigma_s + rgrid.y),
                              c, n),
                      1.f) / cast<float>(sigma_s * sigma_s);
    Func f_grid("f_grid");
    f_grid(x, y, z, c, n) = 0.f;
    f_grid(x, y, lower_bin, c, n) += val * (1.f - w);
    f_grid(x, y, upper_bin, c, n) += val * w;

    // Perform 3D filtering in the grid
    RDom rr(filter_r);
    RDom rs(filter_s);
    Func blur_z("blur_z");
    blur_z(x, y, z, c, n) = 0.f;
    blur_z(x, y, z, c, n) += f_grid(x, y, z + rr.x - filter_r.width() / 2, c, n) *
                          abs(f_filter_r(rr.x));
    Func blur_y("blur_y");
    blur_y(x, y, z, c, n) = 0.f;
    blur_y(x, y, z, c, n) += blur_z(x, y + rs.x - filter_s.width() / 2, z, c, n) *
                          abs(f_filter_s(rs.x));
    Func blur_x("blur_x");
    blur_x(x, y, z, c, n) = 0.f;
    blur_x(x, y, z, c, n) += blur_y(x + rs.x - filter_s.width() / 2, y, z, c, n) *
                          abs(f_filter_s(rs.x));

    // Enclosing voxel
    Expr gx = x / float(sigma_s);
    Expr gy = y / float(sigma_s);
    Expr gz = clamp(f_guide(x, y, n) * cast<float>(sigma_r),
                    0.f,
                    cast<float>(sigma_r));
    Expr fx = cast<int>(floor(gx));
    Expr fy = cast<int>(floor(gy));
    Expr fz = cast<int>(floor(gz));
    Expr cx = fx + 1;
    Expr cy = fy + 1;
    Expr cz = cast<int>(ceil(gz));
    Expr wx = gx - fx;
    Expr wy = gy - fy;
    Expr wz = gz - fz;

    // trilerp
    Func unnormalized_output("unnormalized_output");
    unnormalized_output(x, y, c, n) =
         blur_x(fx, fy, fz, c, n)*(1.f - wx)*(1.f - wy)*(1.f - wz)
       + blur_x(fx, fy, cz, c, n)*(1.f - wx)*(1.f - wy)*(      wz)
       + blur_x(fx, cy, fz, c, n)*(1.f - wx)*(      wy)*(1.f - wz)
       + blur_x(fx, cy, cz, c, n)*(1.f - wx)*(      wy)*(      wz)
       + blur_x(cx, fy, fz, c, n)*(      wx)*(1.f - wy)*(1.f - wz)
       + blur_x(cx, fy, cz, c, n)*(      wx)*(1.f - wy)*(      wz)
       + blur_x(cx, cy, fz, c, n)*(      wx)*(      wy)*(1.f - wz)
       + blur_x(cx, cy, cz, c, n)*(      wx)*(      wy)*(      wz);
    Func output("output");
    output(x, y, c, n) = unnormalized_output(x, y, c, n) /
                      (unnormalized_output(x, y, input.channels(), n) + 1e-4f);

    std::map<std::string, Func> func_map;
    func_map["f_input"]  = f_input;
    func_map["f_guide"]  = f_guide;
    func_map["f_filter_s"] = f_filter_s;
    func_map["f_filter_r"] = f_filter_r;
    func_map["output"] = output;
    return func_map;
}

