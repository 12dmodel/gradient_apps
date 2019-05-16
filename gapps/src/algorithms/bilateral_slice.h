#pragma once

#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), co("co"), n("n");

template <typename Input>
std::map<std::string, Func> bilateral_slice(
        const Input &grid,
        const Input &guide) {
    Func f_grid = BoundaryConditions::repeat_edge(grid);
    Func f_guide = BoundaryConditions::repeat_edge(guide);

    int sigma_s = 32;
    Expr gd = grid.dim(2).extent();

    // Enclosing voxel
    Expr gx = (x+0.5f) / sigma_s;
    Expr gy = (y+0.5f) / sigma_s;
    Expr gz = clamp(f_guide(x, y, n), 0.0f, 1.0f)*gd;

    Expr fx = cast<int>(floor(gx-0.5f));
    Expr fy = cast<int>(floor(gy-0.5f));
    Expr fz = cast<int>(floor(gz-0.5f));

    Expr wx = abs(gx-0.5f - fx);
    Expr wy = abs(gy-0.5f - fy);
    Expr wz = abs(gz-0.5f - fz);

    // Slice the grid
    Func output("output");
    output(x, y, co, n) =
         f_grid(fx  , fy  , fz  , co, n)*(1.f - wx)*(1.f - wy)*(1.f - wz)
       + f_grid(fx  , fy  , fz+1, co, n)*(1.f - wx)*(1.f - wy)*(      wz)
       + f_grid(fx  , fy+1, fz  , co, n)*(1.f - wx)*(      wy)*(1.f - wz)
       + f_grid(fx  , fy+1, fz+1, co, n)*(1.f - wx)*(      wy)*(      wz)
       + f_grid(fx+1, fy  , fz  , co, n)*(      wx)*(1.f - wy)*(1.f - wz)
       + f_grid(fx+1, fy  , fz+1, co, n)*(      wx)*(1.f - wy)*(      wz)
       + f_grid(fx+1, fy+1, fz  , co, n)*(      wx)*(      wy)*(1.f - wz)
       + f_grid(fx+1, fy+1, fz+1, co, n)*(      wx)*(      wy)*(      wz);


    std::map<std::string, Func> func_map;
    func_map["f_grid"]  = f_grid;
    func_map["f_guide"]  = f_guide;
    func_map["output"] = output;
    return func_map;
}

