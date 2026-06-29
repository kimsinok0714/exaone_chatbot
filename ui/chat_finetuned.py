"""
ui/chat_finetuned.py — Stage 3: 파인튜닝 챗봇
사전: 01_finetune.ipynb + 02_merge_model.ipynb 완료
실행: python -m ui.chat_finetuned

📌 이 파일의 역할
- Stage 3에서 만든 병합 모델(models/v1_law/)을 Gradio 챗봇으로 띄운다.
- ui/chat_basic.py 와 구조는 똑같고 **엔진 클래스만 교체** → 분리 설계의 효과를 체감.

🎯 chat_basic.py 와의 차이 (딱 2줄)
    from core.base_engine import BaseEngine       →  from core.finetuned_engine import FinetunedEngine
    engine = BaseEngine()                         →  engine = FinetunedEngine()
"""

# 1️⃣ Gradio + 파인튜닝 엔진 임포트
#    FinetunedEngine은 BaseEngine을 상속받으므로 .chat() 시그니처가 동일
#    → ChatInterface에 그대로 연결 가능
import gradio as gr
from core.finetuned_engine import FinetunedEngine


# 2️⃣ 엔진 인스턴스 생성 (모듈 로드 시 1회)
#    내부 동작:
#    - MERGED_DIR (models/v1_law/) 존재 여부 사전 검증
#    - config.json 유무로 병합 완료 여부 확인
#    - 문제 있으면 FileNotFoundError로 친절한 안내 후 종료
#    - 문제 없으면 BaseEngine._load() 호출 → 4-bit 양자화 로드
engine = FinetunedEngine()


# 3️⃣ ChatInterface 구성 (chat_basic.py와 동일한 패턴)
demo = gr.ChatInterface(
    fn=engine.chat,                           # 리스코프 치환: FinetunedEngine.chat도 동일 시그니처
    title="⚖️ EXAONE 법률 상담 챗봇 (파인튜닝)",  # 도메인에 맞게 변경
    description=engine.get_info_markdown(),   # 오버라이드된 풍부한 메타정보 (학습 데이터, 특징 포함)
    # 4️⃣ 예시 질문 — 모두 법률 도메인 질문으로 구성
    #    파인튜닝 모델의 특화 능력을 시연하는 것이 목적
    #    (chat_basic.py 의 범용 질문과 대조적)
    examples=[
        "임차인이 보증금을 돌려받지 못할 때 어떻게 해야 하나요?",  # 주택임대차
        "근로계약서 없이 일한 경우 임금 체불 시 대처법은?",       # 근로기준
        "주택연금 가입자가 사망하면 어떻게 되나요?",               # 상속/금융
        "교통사고 후 보험사와 합의 시 주의사항은?",                # 보험/손해배상
        "이웃 간 소음 분쟁 법적 대응 방법은?",                     # 생활민원
    ],
)


# 5️⃣ 엔트리 포인트
#    직접 실행 시에만 서버 시작 — 다른 모듈이 import할 때는 서버가 안 뜸
if __name__ == "__main__":
    demo.launch()
    # 💡 팁: 위니브 등 외부 공유용으로는 demo.launch(share=True)