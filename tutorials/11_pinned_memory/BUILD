load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "pinned_memory",
    srcs = ["pinned_memory.cu"],
    deps = ["//tutorials/common:utils"],
    copts = ["-std=c++17"],
)

cc_binary(
    name = "main",
    deps = [":pinned_memory"],
    copts = ["-std=c++17"],
)