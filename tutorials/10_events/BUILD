load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "events",
    srcs = ["events.cu"],
    deps = ["//tutorials/common:utils"],
    copts = ["-std=c++17"],
)

cc_binary(
    name = "main",
    deps = [":events"],
    copts = ["-std=c++17"],
)