#!/bin/bash
# 0. cd into TF folder
cd ../tensorflow

# 1. Remove stale version of native client
echo "Removing stale version of native client ..."
rm -rf native_client

# 2. Move latest native client to folder
echo "Moving latest deepspeech.cc and deepspeech.h code to tf folder ..."
cp -r ../native_client .

# 4. Add latest managed sentencefit.h and sentencefit.cc to native client
echo "Moving 'managed' sentencefit to native client ..."
cp ../sentence-fit/c-codebase/SentenceFit/sentfit/sentencefit.cc native_client
cp ../sentence-fit/c-codebase/SentenceFit/sentfit/SF_GLOBAL_vars.h native_client
cp ../sentence-fit/c-codebase/SentenceFit/sentfit/SentenceFit.h native_client

echo "Moving xtensor and xtl to native client ..."
cp -r ../sentence-fit/c-codebase/SentenceFit/sentfit/xtensor native_client
cp -r ../sentence-fit/c-codebase/SentenceFit/sentfit/xtl native_client

echo "Building c binaries ..."
bazel build --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --config=monolithic -c opt --copt=-O3 --copt="-D_GLIBCXX_USE_CXX11_ABI=0" --define=runtime=tflite //native_client:libdeepspeech.so //native_client:generate_scorer_package

# 8. Clean-up native folder to stop confusion
echo "Cleaning up native client folder to stop confusion"
rm -rf native client
