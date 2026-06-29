"""
config/settings.py — 전체 설정 중앙 관리

📌 이 파일의 역할
- 프로젝트 전체에서 쓰이는 **모든 설정값을 한 곳에 모아둔** 단일 진실 공급원(Single Source of Truth).
- 경로/모델명/하이퍼파라미터를 코드에서 분리 → 실험 재현성과 유지보수성 확보.
- 다른 모든 모듈(core, ui, evaluation, notebooks)이 `from config.settings import ...` 로 이 값들을 참조.

🎯 왜 설정을 분리하는가?
    ❌ 나쁜 예: `model = AutoModel.from_pretrained("LGAI-EXAONE/...")` 를 여러 파일에 하드코딩
    ✅ 좋은 예: 여기 한 줄 바꾸면 전체 프로젝트가 새 모델로 전환

    → 모델 업그레이드, 도메인 교체, 하이퍼파라미터 실험이 매우 쉬워진다.
"""

from pathlib import Path


# ═══════════════════════════════════════════════════════
# 1️⃣ 경로 설정
# ═══════════════════════════════════════════════════════

# 1-1. 프로젝트 루트 자동 탐색
#   __file__        = config/settings.py 의 절대경로
#   .parent         = config/
#   .parent.parent  = project_root/
#   → 어디서 import 하든 항상 동일한 루트를 가리킴
#   (상대경로/cwd 기반 로직보다 훨씬 견고)
ROOT = Path(__file__).parent.parent

# 1-2. 주요 폴더 경로 (ROOT 기준 상대 경로로 정의)
DATA_DIR = ROOT / "data"                             # 학습 데이터 (JSON 등)
MODELS_DIR = ROOT / "models"                         # 어댑터/병합 모델 저장소
EVAL_LOGS_DIR = ROOT / "evaluation" / "logs"         # 평가 결과 JSON 로그

# 1-3. 폴더 자동 생성
#   parents=True:  중간 경로 없어도 함께 생성 (mkdir -p 와 동일)
#   exist_ok=True: 이미 존재해도 에러 안 냄
#   → 이 파일을 import 만 해도 필요한 폴더가 준비됨 (UX 개선)
for d in [DATA_DIR, MODELS_DIR, EVAL_LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════
# 2️⃣ 모델 설정
# ═══════════════════════════════════════════════════════

# 2-1. 베이스 모델 — HuggingFace Hub repo id
#   EXAONE 4.0 1.2B: LG AI Research의 한국어 특화 sLLM
#   ⚠️ Gated 모델이므로 HF 계정에서 라이선스 동의 + HUGGINGFACE_TOKEN 필요
BASE_MODEL_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# 2-2. 파인튜닝 산출물 경로
ADAPTER_DIR = MODELS_DIR / "v1_law_adapter"   # Stage 3-1 결과: LoRA 어댑터 (~20MB)
MERGED_DIR = MODELS_DIR / "v1_law"              # Stage 3-2 결과: 베이스+LoRA 병합 모델 (~2.4GB)

# 💡 버전 관리 팁: v1, v2, v3 ... 으로 실험마다 이름 바꾸면 과거 모델 보존 가능
#    예: MODELS_DIR / "v2_law_3000건"  /  "v3_law_augmented" ...


# ═══════════════════════════════════════════════════════
# 3️⃣ 학습 데이터
# ═══════════════════════════════════════════════════════
# Alpaca 포맷 JSON — [{"instruction":..., "input":..., "output":...}, ...]
TRAIN_DATA_PATH = DATA_DIR / "refined_law_qa.json"


# ═══════════════════════════════════════════════════════
# 4️⃣ 시스템 프롬프트
# ═══════════════════════════════════════════════════════

# 4-1. 기본 프롬프트 — BaseEngine 에서 사용
#   도메인 지정 없이 범용 한국어 어시스턴트
SYSTEM_DEFAULT = "당신은 한국어를 유창하게 사용하는 AI 어시스턴트입니다."

# 4-2. 법률 도메인 프롬프트 — FinetunedEngine + 학습 전처리에서 사용
#   4요소 프롬프트 구조:
#   [1] 역할(Role)      — "대한민국 법률 전문 AI"
#   [2] 원칙(Principle) — 법조항 명시, 결론-근거-절차 순서
#   [3] 제약(Guard)     — 변호사 대체 X (법적 책임 회피)
#   학습 시 모든 샘플에 이 프롬프트가 시스템 메시지로 붙음
#   → 파인튜닝이 이 "페르소나"를 내재화하게 됨
SYSTEM_LEGAL = """당신은 대한민국 법률 전문 AI 어시스턴트입니다.

답변 시 다음 원칙을 따르세요:
1. 관련 법조항을 명시 (예: 민법 제000조)
2. 결론 → 근거 → 실행 절차 순서
3. 변호사 자문을 대체하지 않음을 명시"""


# ═══════════════════════════════════════════════════════
# 5️⃣ 생성 파라미터 (EXAONE 4.0 공식 권장값)
# ═══════════════════════════════════════════════════════
GENERATION_CONFIG = {
    # 최대 생성 토큰 수 — 답변 길이 상한
    "max_new_tokens": 400,

    # 낮은 temperature → 일관성/정확성 우선 (법률 도메인에 적합)
    # 창의적 글쓰기라면 0.7~0.9, 코드 생성은 0.1~0.3
    "temperature": 0.1,

    # nucleus sampling: 누적 확률 상위 90% 토큰에서만 샘플링
    # → 엉뚱한 low-probability 토큰 배제
    "top_p": 0.9,

    # 샘플링 사용 (False면 greedy decoding — 완전 결정적)
    "do_sample": True,

    # 반복 억제: 이미 나온 토큰에 대한 로짓 페널티
    # 1.0 = 없음, 1.1~1.3 = 약한~중간 억제, 1.5+ = 과도한 억제(어색해짐)
    "repetition_penalty": 1.1,
}

# 5-1. 컨텍스트 길이 제한
MAX_CONTEXT_LENGTH = 2048  # 추론 시: 긴 대화 히스토리 슬라이딩 윈도우 상한
MAX_SEQ_LENGTH = 1024      # 학습 시: 토큰 truncation 기준 (VRAM 절약)
# 💡 왜 다른가?
#   - 추론: 사용자 대화가 길어질 수 있음 → 넉넉하게
#   - 학습: 배치 크기와 VRAM 균형 → 짧게 (1024로도 대부분의 Q&A 수용)


# ═══════════════════════════════════════════════════════
# 6️⃣ LoRA 설정
# ═══════════════════════════════════════════════════════
LORA_CONFIG = {
    # 6-1. 저차원 랭크 r — LoRA의 핵심 하이퍼파라미터
    #   r이 클수록: 표현력 ↑, 파라미터 ↑, VRAM ↑, 오버피팅 위험 ↑
    #   r=8:  경량 (소규모 데이터)
    #   r=16: 균형 (이 프로젝트 — 3000건 적정)
    #   r=32: 대용량 (10만건+ 데이터)
    "r": 16,

    # 6-2. LoRA alpha — 스케일링 계수
    #   실효 학습률 ≈ alpha / r
    #   관례: alpha = 2r (여기선 32) → 안정적인 시작점
    "lora_alpha": 32,

    # 6-3. LoRA dropout — 과적합 방지
    #   0.05 ~ 0.1 이 일반적
    "lora_dropout": 0.05,

    # 6-4. bias 학습 여부
    #   "none": bias 학습 X (권장 — 파라미터 절약)
    #   "all":  모든 bias 학습
    #   "lora_only": LoRA 레이어의 bias만
    "bias": "none",

    # 6-5. 타겟 모듈 — LoRA를 어디에 부착할지
    #   Attention (q, k, v, o): 문맥 이해 능력 개선
    #   MLP (gate, up, down):   지식 저장소 개선
    #   모두 포함 = 전체 튜닝에 가장 가까운 효과 (이 프로젝트 선택)
    #
    #   최소 구성: ["q_proj", "v_proj"] (QLoRA 논문 권장)
    #   중간 구성: attention 4개만
    #   최대 구성: 위 7개 전부 (현재)
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",     # Attention projection
        "gate_proj", "up_proj", "down_proj",         # MLP (SwiGLU) projection
    ],
}


# ═══════════════════════════════════════════════════════
# 7️⃣ 학습 설정
# ═══════════════════════════════════════════════════════
TRAIN_CONFIG = {
    # 7-1. 전체 데이터 반복 횟수
    #   3000건 × 2 epoch = 6000 샘플 학습
    #   너무 많으면 오버피팅 (도메인 지식은 늘지만 범용성은 잃음)
    "num_train_epochs": 2,

    # 7-2. GPU당 실제 배치 크기
    #   VRAM 따라 조정 — 2가 RTX 3060/4060 기준 안전값
    "per_device_train_batch_size": 2,

    # 7-3. 그래디언트 누적
    #   실효 배치 = 2 × 4 = 8
    #   VRAM 부족 시 여기만 늘리면 "배치를 키운 효과" 얻음
    "gradient_accumulation_steps": 4,

    # 7-4. 학습률
    #   LoRA 표준: 1e-4 ~ 3e-4
    #   Full fine-tuning 대비 10~100배 큰 값 (LoRA 파라미터가 소수라서)
    "learning_rate": 2e-4,

    # 7-5. 스케줄러 — cosine 감소
    #   warmup 후 코사인 곡선으로 부드럽게 0까지 감소
    #   "linear"보다 일반적으로 더 안정적
    "lr_scheduler_type": "cosine",

    # 7-6. 초기 워밍업 스텝
    #   처음 5스텝 동안 LR을 0에서 설정값까지 linear 증가
    #   초기 gradient 폭주 방지
    "warmup_steps": 5,

    # 7-7. 옵티마이저
    #   paged_adamw_8bit: bitsandbytes의 8-bit Adam
    #   - state를 8-bit로 저장 → VRAM ~75% 절감
    #   - "paged": OOM 직전 CPU RAM으로 자동 스왑
    "optim": "paged_adamw_8bit",

    # 7-8. bfloat16 혼합정밀도
    #   Ampere (RTX 30xx) 이상 GPU에서 지원
    #   fp16보다 안정적 (NaN 없음), 속도는 동일
    "bf16": True,

    # 7-9. 로그 출력 간격
    #   10스텝마다 loss 출력 → 학습 진행 모니터링
    "logging_steps": 10,
}


# ═══════════════════════════════════════════════════════
# 8️⃣ 설정 확인 유틸리티
# ═══════════════════════════════════════════════════════
def print_settings():
    """설정 확인 출력
    
    노트북 상단에서 호출해 현재 실험 설정을 한눈에 파악.
    실험 재현성을 위해 **실행 기록과 함께 남기는 습관** 권장.
    """
    print("=" * 50)
    print("📋 EXAONE Chatbot 설정")
    print("=" * 50)
    print(f"베이스 모델:  {BASE_MODEL_NAME}")
    print(f"학습 데이터:  {TRAIN_DATA_PATH}")
    print(f"어댑터 경로:  {ADAPTER_DIR}")
    print(f"병합 경로:    {MERGED_DIR}")
    print(f"평가 로그:    {EVAL_LOGS_DIR}")
    print("=" * 50)


# 9️⃣ 단독 실행 — 설정 검증용
#    `python -m config.settings` 실행 시
#    경로 자동 생성 + 설정 요약 출력
#    → 환경 설정이 제대로 되어 있는지 빠른 점검
if __name__ == "__main__":
    print_settings()
