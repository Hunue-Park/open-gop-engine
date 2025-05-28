import numpy as np
import soundfile as sf
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from tokenizers import Tokenizer
from collections import defaultdict
import logging
import torch
from dtw import dtw
import math

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Wav2VecCTCOnnxCore:
    """
    ONNX Runtime based Wav2Vec2 CTC inference engine using a quantized ONNX model.
    Includes fallback for prototype matrix in either orientation.
    """

    def __init__(
        self,
        onnx_model_path: str,
        tokenizer_path: str,
        device: str = "CPU"
    ):
        self.weight_norm_mid = 50
        self.weight_norm_steepness = 0.2
        # 1) session & model load
        providers = ["CPUExecutionProvider"] if device.upper() == "CPU" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.onnx_model = onnx.load(onnx_model_path)

        # 2) tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # 3) I/O names
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name = inputs[0].name
        self.hidden_name = outputs[0].name
        self.logits_name = outputs[1].name
        logger.debug("Model IO names: input=%s, hidden=%s, logits=%s",
                     self.input_name, self.hidden_name, self.logits_name)

        # 4) infer hidden_dim & vocab_size from output shapes
        hidden_dim = None
        vocab_size = None
        for output in self.onnx_model.graph.output:
            dims = output.type.tensor_type.shape.dim
            if output.name == self.hidden_name:
                hidden_dim = dims[2].dim_value
            elif output.name == self.logits_name:
                vocab_size = dims[2].dim_value
        if hidden_dim is None or vocab_size is None:
            raise RuntimeError("Could not determine hidden_dim or vocab_size from model outputs.")

        # 5) load & dequantize lm_head weight (prototype matrix)
        proto = None
        # try to find quantized weight + scale + zero_point
        quant_init = None
        for init in self.onnx_model.graph.initializer:
            arr = numpy_helper.to_array(init)
            # look for (hidden_dim, vocab_size) quantized weight
            if arr.ndim == 2 and arr.shape == (hidden_dim, vocab_size) and init.name.endswith("_quantized"):
                quant_init = init
                quant_arr = arr.astype(np.float32)
                break

        if quant_init is not None:
            base = quant_init.name[:-len("_quantized")]
            # find scale and zero_point of shape (vocab_size,)
            scale_arr = numpy_helper.to_array(
                next(i for i in self.onnx_model.graph.initializer if i.name == base + "_scale")
            ).astype(np.float32)
            zp_arr = numpy_helper.to_array(
                next(i for i in self.onnx_model.graph.initializer if i.name == base + "_zero_point")
            ).astype(np.float32)
            # dequantize: (Q - zp) * scale
            dequant = (quant_arr - zp_arr) * scale_arr
            # transpose => (vocab_size, hidden_dim)
            proto = dequant.T
        else:
            # fallback: scan initializers for exact or transposed orientation
            for init in self.onnx_model.graph.initializer:
                arr = numpy_helper.to_array(init)
                if arr.ndim == 2 and arr.shape == (vocab_size, hidden_dim):
                    proto = arr
                    break
            if proto is None:
                for init in self.onnx_model.graph.initializer:
                    arr = numpy_helper.to_array(init)
                    if arr.ndim == 2 and arr.shape == (hidden_dim, vocab_size):
                        proto = arr.T
                        break

        if proto is None:
            raise RuntimeError("Prototype matrix (lm_head weight) not found in any initializer.")
        self.prototype_matrix = proto

        logger.debug("Loaded prototype_matrix of shape %s", self.prototype_matrix.shape)

    def dtw_align(self, X, Y):
        alignment = dtw(X, Y, keep_internals=True, step_pattern="asymmetricP1")
        return alignment.index1, alignment.index2

    def transcribe(self, audio_path: str, raw_ids: list) -> str:
        # logger.debug("Transcribing %s", audio_path)
        # audio = self.load_audio(audio_path)
        # logits = self.extract_logits(audio)
        # logits.shape == (T, V) 이어야 함

        # raw_ids = logits.argmax(axis=1).tolist()
        logger.debug("Raw Greedy IDs: %s", raw_ids)

        # special tokens 문자열로 직접 지정
        blank_token = "|"      # CTC blank
        pad_token   = "[PAD]"
        unk_token   = "[UNK]"

        # ID 조회
        blank_id = self.tokenizer.token_to_id(blank_token)
        pad_id   = self.tokenizer.token_to_id(pad_token)
        unk_id   = self.tokenizer.token_to_id(unk_token)

        dedup_ids = []
        prev = None
        for idx in raw_ids:
            # special token이면 prev도 초기화해서 다음 non‑special이 반드시 append 되도록
            if idx in (blank_id, pad_id, unk_id):
                prev = None
                continue

            # non‑special인데 연속 중복이면 skip
            if idx == prev:
                continue

            dedup_ids.append(idx)
            prev = idx

        logger.debug("CTC‑decoded IDs: %s", dedup_ids)

        # 최종 디코딩
        text = self.tokenizer.decode(dedup_ids)
        logger.debug("Decoded text: %s", text)
        return text
    
    def sigmoid_weight(self, score, mid=35, steepness=0.2):
        # 0.5~1.5 범위의 가중치 생성
        return 0.5 + 1.0 / (1 + math.exp(-steepness * (score - mid)))

    def weighted_avg_with_sigmoid(self, syllables, mid=35, steepness=0.2):
        total_weighted_score = 0
        total_weight = 0
        
        for syl, score in syllables:
            if syl == '|':  # 구분자 건너뛰기
                continue
            weight = self.sigmoid_weight(score, mid, steepness)
            total_weighted_score += score * weight
            total_weight += weight
        
        raw_score = total_weighted_score / total_weight if total_weight > 0 else 0
        return min(raw_score, 100)

    def group_words_sigmoid(self, syllable_scores):
        words = []
        current_word = []
        
        for i, (syl, score) in enumerate(syllable_scores):
            if syl == '|':
                if current_word:
                    word_text = ''.join(s[0] for s in current_word)
                    word_score = round(self.weighted_avg_with_sigmoid(current_word, self.weight_norm_mid, self.weight_norm_steepness))
                    words.append({"word": word_text, "scores": {"pronunciation": word_score}})
                    current_word = []
            else:
                current_word.append((syl, score))
        
        # 마지막 단어 처리
        if current_word:
            word_text = ''.join(s[0] for s in current_word)
            word_score = round(self.weighted_avg_with_sigmoid(current_word, self.weight_norm_mid, self.weight_norm_steepness))
            words.append({"word": word_text, "scores": {"pronunciation": word_score}})
        
        return words

    def calculate_gop_from_tensor(self, audio_tensor: torch.Tensor, text: str, eps: float = 1e-8) -> dict:
        """
        전처리된 오디오 텐서에서 직접 GOP 계산
        
        Args:
            audio_tensor: 전처리된 오디오 텐서 [1, T]
            text: 평가할 텍스트
            eps: 수치 안정성을 위한 작은 값
            
        Returns:
            dict: GOP 평가 결과
        """
        # 텐서를 numpy 배열로 변환
        input_np = audio_tensor.numpy()
        
        # 2) run ONNX to get hidden & logits
        hidden_np, logits_np = self.session.run(
            [self.hidden_name, self.logits_name],
            {self.input_name: input_np}
        )
        
        # remove batch dim
        X      = hidden_np[0]   # shape (T, D)
        logits = logits_np[0]   # shape (T, V)

        # 3) temperature‐scaled softmax → probs
        scaled     = logits
        exp_logits = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs      = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # 4) tokenize
        text = text.replace(" ", "|")
        token_ids = self.tokenizer.encode(text).ids

        V = self.prototype_matrix.shape[0]
        blank_id = self.tokenizer.token_to_id("|")
        safe_ids = [tid if 0 <= tid < V else blank_id for tid in token_ids]

        # 5) expand prototypes & DTW
        proto = self.prototype_matrix[safe_ids]  # (M, D)
        T, M   = X.shape[0], len(safe_ids)
        avg    = max(1, T // M)
        Yexp   = np.repeat(proto, avg, axis=0)  # (M*avg, D)
        pX, pYexp = self.dtw_align(X, Yexp)
        pY    = [y // avg for y in pYexp]

        # 6) collect frames per token
        frames = defaultdict(list)
        for f, t in zip(pX, pY):
            frames[t].append(f)

        # 7) per‐token log‐prob scores
        tok_scores = []
        for idx, tid in enumerate(safe_ids):
            tok = self.tokenizer.id_to_token(tid)
            frs = frames.get(idx, [])
            if frs:
                ps    = probs[frs, tid]
                score = float(np.mean(np.log(ps + eps)))
            else:
                score = float(-np.inf)
            tok_scores.append((tok, score))

        # 8) normalize to [0,100]
        raw  = np.array([s for _, s in tok_scores], dtype=np.float32)
        mask = np.isfinite(raw)
        if mask.any():
            mn   = raw[mask].min()
            mx   = raw[mask].max()
            span = mx - mn if mx > mn else eps
            norm = [(t, (s - mn) / span * 100.0 if np.isfinite(s) else 0.0)
                    for t, s in tok_scores]
        else:
            norm = [(t, 0.0) for t, _ in tok_scores]

        # 9) group into words
        words = self.group_words_sigmoid(norm)

        overall = (
            round(sum(w["scores"]["pronunciation"] for w in words) / len(words), 1)
            if words else 0.0
        )
        return {"overall": overall, "pronunciation": overall, "words": words}

    def calculate_gop_with_context(self, audio_tensor: torch.Tensor, target_text: str, 
                                   context_before: str = "", context_after: str = "", 
                                   target_index: int = None) -> dict:
        """
        컨텍스트를 고려하여 특정 블록의 GOP 계산
        
        Args:
            audio_tensor: 전처리된 오디오 텐서
            target_text: 평가할 대상 텍스트
            context_before: 대상 전의 컨텍스트
            context_after: 대상 후의 컨텍스트
            target_index: 전체 텍스트에서 대상의 인덱스 (없으면 자동 계산)
            
        Returns:
            dict: 대상 블록에 대한 GOP 평가 결과
        """
        # 컨텍스트를 포함한 전체 텍스트
        full_text = f"{context_before} {target_text} {context_after}".strip()
        
        # 대상 텍스트의 인덱스 계산
        if target_index is None:
            # context_before의 단어 수를 세어 target의 시작 인덱스 결정
            words_before = len([w for w in context_before.split() if w]) if context_before else 0
            target_index = words_before
        
        # 대상 텍스트의 단어 수 계산
        target_word_count = len([w for w in target_text.split() if w])
        
        # 전체 텍스트로 GOP 계산
        result = self.calculate_gop_from_tensor(audio_tensor, full_text)
        
        # 모든 단어가 있는지 확인
        if not result["words"] or len(result["words"]) <= target_index:
            # 전체 텍스트 처리에 실패한 경우, 대상 텍스트만으로 시도
            fallback_result = self.calculate_gop_from_tensor(audio_tensor, target_text)
            return fallback_result
        
        # target_index 위치의 단어들에 해당하는 결과 추출
        # 인덱스 범위 유효성 검사
        end_index = min(target_index + target_word_count, len(result["words"]))
        target_words = result["words"][target_index:end_index]
        
        # 대상 블록에 대한 결과 생성
        if target_words:
            target_score = sum(w["scores"]["pronunciation"] for w in target_words) / len(target_words)
        else:
            target_score = 0.0
        
        target_result = {
            "overall": round(target_score, 1),
            "pronunciation": round(target_score, 1),
            "words": target_words
        }
        
        return target_result

