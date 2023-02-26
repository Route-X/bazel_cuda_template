load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "host_device_functions",
    srcs = ["host_device_functions.cu"],
)

cc_binary(
    name = "main",
    deps = [":host_device_functions"]
)