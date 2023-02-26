load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "grid_configuration",
    srcs = ["grid_configuration.cu"],
)

cc_binary(
    name = "main",
    deps = [":grid_configuration"]
)