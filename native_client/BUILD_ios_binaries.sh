#!/bin/bash
# 0. cd into TF folder
cd ../tensorflow

# 1. Remove stale version of native client
echo "Removing stale version of native client ..."
rm -rf native_client

# 2. Move latest native client to folder
echo "Moving latest deepspeech.cc and deepspeech.h code to tf folder ..."
cp -r ../native_client .

# 3. Add latest managed sentencefit.h and sentencefit.cc to native client
echo "Moving 'managed' sentencefit to native client ..."
cp ../sentence-fit/c-codebase/SentenceFit/sentfit/sentencefit.cc native_client
cp ../sentence-fit/c-codebase/SentenceFit/sentfit/SentenceFit.h native_client
cp ../sentence-fit/c-codebase/SentenceFit/sentfit/SF_GLOBAL_vars.h native_client
cp -r ../sentence-fit/c-codebase/SentenceFit/sentfit/xtensor native_client
cp -r ../sentence-fit/c-codebase/SentenceFit/sentfit/xtl native_client

# 4. Build binaries
# bazel clean
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` - 1 : Compile x86_64 libraries needed for sim in Xcode [-h]"
  echo "Usage: `basename $0` - 2 : Compile arm64 libraries needed for on-device sim - good for on-device metrics [-h]"
  exit 0
fi

build_config=ios_x86_64
if [ "$1" -eq 1 ]
then
    build_config=ios_x86_64
fi 

if [ "$1" -eq 2 ]
then
    build_config=ios_arm64
fi 

echo "Building $build_config binaries ..."
bazel build --verbose_failures --config=$build_config  --workspace_status_command="bash native_client/bazel_workspace_status_cmd.sh" --apple_bitcode=embedded --copt=-fembed-bitcode --config=monolithic -c opt //native_client:deepspeech_ios --define=runtime=tflite --copt=-DTFLITE_WITH_RUY_GEMV

# 6. Remove previous framework/binary
echo "Remove previous compiled ios framework ..."
rm -rf ~/dev/ios-tutu/ios-asr/deepspeech_ios.framework

# 7. Add latest build to deepspeech project
echo "Adding latest build to xcode project ..."
cp -r bazel-bin/native_client/deepspeech_ios_archive-root/deepspeech_ios.framework ../../ios-asr