"""
core/finetuned_engine.py — 파인튜닝 EXAONE 엔진
BaseEngine 상속 — 병합된 로컬 모델 사용

📌 이 파일의 역할
- 파인튜닝(Stage 3) 결과물인 병합 모델(models/v1_law/)을 로드하는 전용 엔진
- BaseEngine을 **상속**해 중복 코드 0줄 — 경로와 프롬프트만 바꿔서 재사용
- 병합 모델 유무를 **사전 검증**해 초보 사용자 친화적인 에러 메시지 제공

🎯 상속 구조의 핵심
    BaseEngine (모든 로드/추론 로직)
        └── FinetunedEngine (경로/라벨/프롬프트만 특화)

→ Gradio UI 코드는 BaseEngine이든 FinetunedEngine이든 **동일한 인터페이스**로 사용 가능
"""

# 1️⃣ 경로 검증용 pathlib + 부모 클래스 + 설정값 임포트
from pathlib import Path
from core.base_engine import BaseEngine
from config.settings import MERGED_DIR, SYSTEM_LEGAL
#                            ↑              ↑
#                            │              └─ 법률 도메인 시스템 프롬프트
#                            └─ 병합 모델 기본 경로 (models/v1_law/)


class FinetunedEngine(BaseEngine):
    """
    파인튜닝된 EXAONE 모델 엔진.

    - 병합 모델(models/v1_law/) 로드
    - 법률 특화 시스템 프롬프트 기본 사용
    """

    # 2️⃣ 생성자 — BaseEngine과 시그니처를 동일하게 유지
    #    (UI 코드에서 engine 교체만으로 동작하도록)
    def __init__(self, model_path=None, system_prompt=None, label=None):
        # 2-1. 기본값 폴백 패턴
        #   model_path=None 이면 settings의 MERGED_DIR 사용
        #   → 일반 사용자는 인자 없이 FinetunedEngine()만 해도 됨
        model_path = str(model_path or MERGED_DIR)
        system_prompt = system_prompt or SYSTEM_LEGAL
        label = label or "EXAONE 4.0 법률 특화 (3,000건 학습)"

        # 2-2. 모델 폴더 존재 확인 (사전 검증 #1)
        #   Stage 3 노트북을 아직 실행하지 않았다면 이 시점에서 친절하게 안내
        #   → BaseEngine.__init__()으로 들어가면 HuggingFace 캐시 관련 혼란스러운
        #      에러가 뜨므로, 그 전에 가로채는 것이 UX상 중요
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"\n❌ 병합 모델 폴더 없음: {model_path}\n"
                f"   해결: notebooks/01_finetune.ipynb → 02_merge_model.ipynb 실행\n"
            )

        # 2-3. 병합 완료 여부 확인 (사전 검증 #2)
        #   config.json 은 save_pretrained() 완료 시점에 마지막으로 생성되는 파일 중 하나
        #   → 있으면 병합이 정상 완료된 상태라고 간주하는 간단한 휴리스틱
        if not (Path(model_path) / "config.json").exists():
            raise FileNotFoundError(
                f"\n❌ 병합이 완료되지 않음 (config.json 없음): {model_path}\n"
                f"   해결: notebooks/02_merge_model.ipynb 재실행\n"
            )

        # 2-4. 부모 클래스 __init__ 호출
        #   BaseEngine의 _load()가 여기서 실행됨 (토크나이저 + 4-bit 양자화 로드)
        #   ⭐ 이 한 줄 덕분에 모델 로드/추론 로직을 단 한 줄도 다시 쓸 필요가 없음
        super().__init__(
            model_path=model_path,
            system_prompt=system_prompt,
            label=label,
        )

    # 3️⃣ UI 설명 메시지 오버라이드
    #    BaseEngine보다 풍부한 메타정보를 제공 — 학습 데이터/특징까지 표시
    def get_info_markdown(self):
        return (
            f"**모델**: {self.label}\n\n"
            f"**경로**: `{self.model_path}`\n\n"
            f"**학습 데이터**: 한국 법률 QA 3,000건\n\n"
            f"**특징**: 법조항 인용, 구조화된 답변"
        )


# 4️⃣ 단독 실행 테스트
#    `python -m core.finetuned_engine` 으로 모델 로드부터 첫 응답까지 한 번에 검증
if __name__ == "__main__":
    # 4-1. try/except로 감싸 FileNotFoundError 를 예쁘게 처리
    #   (Stage 3을 건너뛴 사용자를 위한 배려)
    try:
        engine = FinetunedEngine()
        print(engine.get_info_markdown())

        q = "임차인이 보증금을 돌려받지 못할 때 어떻게 해야 하나요?"
        print(f"\n[Q] {q}")
        print(f"[A] {engine.chat(q)}")
    except FileNotFoundError as e:
        # Stage 3 노트북 실행 안내 메시지를 그대로 출력
        print(e)