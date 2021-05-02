make bindings-clean
pip uninstall deepspeech -y
make bindings
pip install dist/deepspeech-0.9.1-cp37-cp37m-macosx_10_9_x86_64.whl
python client.py --model optimized_model.tflite --scorer lm.scorer --audio 172.wav
