load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "memory_basics",
    srcs = ["memory_basics.cu"],
)

cc_binary(
    name = "main",
    deps = [":memory_basics"]
)