load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "synchronization",
    srcs = ["synchronization.cu"],
    deps = ["//tutorials/common:utils"],
)

cc_binary(
    name = "main",
    deps = [":synchronization"]
)