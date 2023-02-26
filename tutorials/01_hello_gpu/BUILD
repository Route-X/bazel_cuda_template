load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "hello_gpu",
    srcs = ["hello_gpu.cu"],
)

cc_binary(
    name = "main",
    deps = [":hello_gpu"]
)