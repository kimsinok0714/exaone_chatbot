"""
core/base_engine.py — 기본 EXAONE 엔진

📌 이 파일의 역할
- HuggingFace에 공개된 EXAONE 4.0 모델을 **4-bit 양자화**로 로드한다.
- Gradio ChatInterface가 요구하는 `chat(message, history) -> str` 인터페이스를 제공한다.
- UI(`ui/chat_basic.py`)와 모델을 분리(관심사 분리)하기 위한 엔진 레이어.
"""

# 1️⃣ 핵심 라이브러리
import torch  # PyTorch: dtype, inference_mode, CUDA 메모리 조회 등
from transformers import (
    AutoTokenizer,          # 체크포인트에 맞는 토크나이저 자동 로드
    AutoModelForCausalLM,   # Causal LM(다음 토큰 예측) 아키텍처 자동 로드
    BitsAndBytesConfig,     # 4/8-bit 양자화 설정 객체
)

# 2️⃣ 프로젝트 설정값(하드코딩 금지 원칙) — config/settings.py 에서 주입
#    - BASE_MODEL_NAME: HuggingFace repo id 또는 로컬 경로
#    - SYSTEM_DEFAULT:  시스템 프롬프트(모델의 페르소나/정책)
#    - GENERATION_CONFIG: max_new_tokens, temperature 등 generate() 인자
#    - MAX_CONTEXT_LENGTH: 토큰 길이 초과 방어 임계값
from config.settings import (
    BASE_MODEL_NAME, SYSTEM_DEFAULT,
    GENERATION_CONFIG, MAX_CONTEXT_LENGTH,
)


class BaseEngine:
    """HuggingFace 공개 EXAONE 4.0 모델 엔진"""

    # 3️⃣ 생성자 — 설정 주입 + 즉시 모델 로드
    def __init__(self, model_path=None, system_prompt=None, label=None):
        # 인자로 받은 값이 있으면 사용, 없으면 settings의 기본값으로 폴백
        # → 동일 클래스를 base/tuned/다른 경로로 재사용 가능 (DI 패턴)
        self.model_path = model_path or BASE_MODEL_NAME
        self.system_prompt = system_prompt or SYSTEM_DEFAULT
        self.label = label or "EXAONE 4.0 (순정)"

        # 모델/토크나이저는 _load()에서 주입
        self.model = None
        self.tokenizer = None
        self._load()  # ⚠️ 생성자에서 바로 로드 — 앱 시작 시 1회만 돌도록 설계

    # 4️⃣ 모델 & 토크나이저 로드
    def _load(self):
        print(f"\n🔄 모델 로드: {self.label}")
        print(f"   경로: {self.model_path}")

        # 4-1. QLoRA 스타일 4-bit 양자화 설정
        #   - nf4: 정규분포에 최적화된 4-bit 포맷(논문 기준 권장)
        #   - compute_dtype=bfloat16: 행렬 연산은 bf16로 수행(정확도 유지)
        #   - double_quant: 양자화 상수를 한 번 더 양자화 → VRAM 추가 절감
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # 4-2. 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # pad_token이 없는 모델(대부분의 LLM) 대비 — eos로 대체하는 관용 패턴
        # batch 생성/어텐션 마스크 계산 시 pad_token이 반드시 필요함
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-3. 모델 로드 (4-bit 양자화 적용 + 자동 디바이스 배치)
        #   device_map="auto": GPU가 있으면 GPU, 부족하면 CPU로 자동 분산
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb,
            device_map="auto",
        )
        self.model.eval()                 # 추론 모드: dropout OFF, batchnorm 고정
        self.model.config.use_cache = True  # KV 캐시 사용 → generate 속도 향상
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # 4-4. 사용 중인 VRAM을 사람 친화적 단위(GB)로 출력
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"✅ 완료 (VRAM: {vram:.2f}GB)\n")

    # 5️⃣ Gradio history → ChatML messages 변환
    #    Gradio 버전에 따라 히스토리 포맷이 다르기 때문에 양쪽 모두 지원한다.
    def _build_messages(self, message, history):
        """
        Gradio history를 ChatML로 변환.
        4.x(tuple) / 5.x(dict) 양쪽 호환.
        """
        # 5-1. 시스템 프롬프트는 항상 맨 앞에 1회
        messages = [{"role": "system", "content": self.system_prompt}]

        # 5-2. 과거 대화 이력을 순서대로 추가
        for turn in history or []:
            # ── Gradio 5.x: messages 포맷 ({"role": "user"/"assistant", "content": ...})
            if isinstance(turn, dict) and "role" in turn:
                content = turn.get("content", "")
                # 멀티모달 등에서 content가 list인 경우 텍스트만 이어붙임
                if isinstance(content, list):
                    content = "".join(
                        i.get("text", "") for i in content
                        if isinstance(i, dict)
                    )
                messages.append({"role": turn["role"], "content": content})

            # ── Gradio 4.x: tuples 포맷 ((user, bot))
            elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                u, b = turn
                if u:
                    messages.append({"role": "user", "content": u})
                if b:
                    messages.append({"role": "assistant", "content": b})

        # 5-3. 이번 턴의 사용자 입력을 마지막에 추가
        messages.append({"role": "user", "content": message})
        return messages

    # 6️⃣ 추론(generate) — Gradio ChatInterface가 매 턴마다 호출하는 메서드
    def chat(self, message, history=None):
        """Gradio ChatInterface용 추론"""
        # 6-1. 히스토리 + 이번 입력을 ChatML messages로 정리
        messages = self._build_messages(message, history)

        # 6-2. 토큰화 — ⭐ 노바코프 권장 "2단계 안정 패턴"
        #   (1) apply_chat_template(..., tokenize=False) → 순수 문자열 프롬프트
        #   (2) tokenizer(prompt, return_tensors="pt")   → 텐서화
        #   한 번에 하면 EXAONE에서 dtype/device 충돌이 나는 경우가 있어 분리.
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # 모델이 이어 쓸 "assistant:" 프리픽스 추가
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]

        # 6-3. 컨텍스트 길이 초과 방어
        #   최근 MAX_CONTEXT_LENGTH 토큰만 남겨 슬라이딩 윈도우처럼 동작.
        #   장시간 대화에서 OOM/속도 저하를 예방하는 단순하지만 중요한 안전장치.
        if input_ids.shape[1] > MAX_CONTEXT_LENGTH:
            input_ids = input_ids[:, -MAX_CONTEXT_LENGTH:]

        # 6-4. 텍스트 생성
        #   inference_mode()는 no_grad()보다 한층 더 엄격한 추론 전용 컨텍스트
        #   (오토그래드 비활성 + 버전 카운터 비활성 → 약간 더 빠름)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                **GENERATION_CONFIG,  # max_new_tokens, temperature, top_p 등
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 6-5. 출력에서 "입력 프롬프트" 부분을 잘라내고 새로 생성된 토큰만 디코드
        #   outputs[0]는 [입력 토큰 ... | 생성 토큰 ...] 이므로 input 길이 이후를 슬라이스.
        gen = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # 7️⃣ Gradio description 용 메타정보 — 마크다운 반환
    def get_info_markdown(self):
        return f"**모델**: {self.label}\n\n**경로**: `{self.model_path}`"


# 8️⃣ 단독 실행 시 빠른 sanity check
#    `python -m core.base_engine` 로 실행하면
#    모델이 정상 로드되고 한 턴 대화가 되는지 즉시 확인 가능.
if __name__ == "__main__":
    engine = BaseEngine()
    print(engine.get_info_markdown())
    q = "안녕하세요! 자기소개 해주세요."
    print(f"\n[질문] {q}")
    print(f"[응답] {engine.chat(q)}")