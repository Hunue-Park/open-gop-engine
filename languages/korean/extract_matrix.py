# extract_matrix.py
import numpy as np
import sys
import os
from realtime_engine_ko.w2v_onnx_core import Wav2VecCTCOnnxCore

def extract_prototype_matrix(model_path, tokenizer_path, output_path):
    # 모델 로드
    print(f"모델 로딩 중: {model_path}")
    model = Wav2VecCTCOnnxCore(
        onnx_model_path=model_path,
        tokenizer_path=tokenizer_path
    )
    
    # 프로토타입 매트릭스 추출
    prototype_matrix = model.prototype_matrix
    shape = prototype_matrix.shape
    print(f"매트릭스 추출 완료: 형태={shape}")
    
    # 파일 경로를 명시적으로 지정
    shape_file = os.path.join(os.path.dirname(output_path), "matrix_shape.txt")
    matrix_file = os.path.join(os.path.dirname(output_path), "prototype_matrix.bin")

    # 형태 정보를 별도로 저장
    with open(shape_file, "w") as f:
        f.write(f"{shape[0]} {shape[1]}")

    # 데이터를 바이너리로 저장
    prototype_matrix.tofile(matrix_file)
    print(f"매트릭스 저장 완료: 형태={shape_file}, 데이터={matrix_file}")
    
    # 간단한 통계 정보 출력
    print(f"통계: 최소값={prototype_matrix.min()}, 최대값={prototype_matrix.max()}")
    print(f"첫 5x5 값:\n{prototype_matrix[:5, :5]}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("사용법: python extract_matrix.py <모델_경로> <토크나이저_경로> <출력_경로>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    tokenizer_path = sys.argv[2]
    output_path = sys.argv[3]
    
    extract_prototype_matrix(model_path, tokenizer_path, output_path)