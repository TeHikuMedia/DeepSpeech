#!/bin/bash
set -ex

# This script will be run bazel when building process starts to
# generate key-value information that represents the status of the
# workspace. The output should be like
#
# KEY1 VALUE1
# KEY2 VALUE2
#
# Keys starting with STABLE_ cause dependent rules to be re-run when their value
# changes.
#
# If the script exits with non-zero code, it's considered as a failure
# and the output will be discarded.

tf_git_rev=v2.3.0-6-g23ad988fcd
echo "STABLE_TF_GIT_VERSION ${tf_git_rev}"

# use this trick to be able to use the script from anywhere
pushd $(dirname "$0")
ds_git_rev=v0.10.0-alpha.3-26-g962a117f
echo "STABLE_DS_GIT_VERSION ${ds_git_rev}"
ds_version=0.10.0-alpha.3
echo "STABLE_DS_VERSION ${ds_version}"
ds_graph_version=6
echo "STABLE_DS_GRAPH_VERSION ${ds_graph_version}"
popd
