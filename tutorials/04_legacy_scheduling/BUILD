load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "legacy_scheduling",
    srcs = ["legacy_scheduling.cu"],
    deps = ["//tutorials/common:utils"],
)

cc_binary(
    name = "main",
    deps = [":legacy_scheduling"]
)