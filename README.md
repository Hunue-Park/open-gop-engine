# Speech Recognition Engine Library

- This speech recognition engine is implemented using Facebook's Wav2Vec 2.0 model.
- Uses quantized onnx models for on-device usage.

## Currently Supported Languages

- Korean

## Implementation Methods

- Implemented as a Python module for real-time speech recognition.
- Implemented as a C++ module for speech recognition.

## Usage Instructions

- Python Module Usage
  - In the languages/korean/python/realtime_engine_ko folder
  - pip install -e .
  - For module usage instructions, please refer to the languages/korean/python/realtime_engine_ko/blueprint.md file.
- C++ Module Usage
  - In the languages/korean/cpp folder
  - mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && make
  - After building, you can implement the speech recognition engine using the generated .a file and .h file.

## Models Used

- Korean Model (Please refer to Hugging Face)
  - languages/korean/models/wav2vec2_ctc_dynamic.onnx
  - languages/korean/models/tokenizer.json

## How to run example python application

- cd languages/korean/python && pip install -e .
- cd languages/korean/example/python-audio-app && python audio_app.py
- python audio_app.py
- follow the instructions to record audio and see the result.
