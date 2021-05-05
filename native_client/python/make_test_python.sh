make bindings-clean
pip uninstall deepspeech -y
make bindings
pip install dist/*.whl
python client.py --model optimized_model.tflite --scorer lm.scorer --audio 172.wav
