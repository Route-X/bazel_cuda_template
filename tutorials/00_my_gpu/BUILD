load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "my_gpu",
    srcs = ["my_gpu.cu"],
)

cc_binary(
    name = "main",
    deps = [":my_gpu"]
)