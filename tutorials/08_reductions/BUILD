load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "reductions",
    srcs = ["reductions.cu"],
    deps = ["//tutorials/common:utils"],
    copts = ["-std=c++17"],
)

cc_binary(
    name = "main",
    deps = [":reductions"],
    copts = ["-std=c++17"],
)