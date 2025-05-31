# languages/korean/python/setup.py
from setuptools import setup, find_packages

setup(
    name="realtime_engine_ko",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "onnxruntime",
        "torch",
        "onnx",
        "tokenizers",
        "dtw-python"
    ],
    description="한국어 실시간 음성 평가 엔진",
    author="OpenGOP Team",
)