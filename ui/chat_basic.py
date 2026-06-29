"""
ui/chat_basic.py — Stage 2: 기본 EXAONE 챗봇
실행: python -m ui.chat_basic

📌 이 파일의 역할
- Gradio의 ChatInterface를 이용해 EXAONE 모델과 대화하는 웹 UI를 띄운다.
- 모델 로딩/추론 로직은 core.base_engine.BaseEngine 에 위임(관심사 분리)하고,
  이 파일은 "UI 조립"만 담당한다.
"""

# 1️⃣ Gradio: 파이썬 함수를 즉시 웹 UI로 감싸주는 라이브러리
#    ChatInterface는 fn(메시지, 히스토리) → 응답 형태의 함수를 받아
#    ChatGPT 스타일 대화창을 자동으로 만들어 준다.
import gradio as gr

# 2️⃣ 모델 추론 엔진 임포트
#    BaseEngine 내부에서 EXAONE 토크나이저/모델 로드, 프롬프트 템플릿 적용,
#    generate() 호출 등 "무거운 일"을 수행한다.
from core.base_engine import BaseEngine


# 3️⃣ 엔진 인스턴스 생성 (모듈 로드 시점에 1회만 실행)
#    - 앱 시작 시 모델을 미리 메모리에 올려두고,
#      이후 모든 대화 요청에서 이 하나의 인스턴스를 재사용한다.
#    - 매 요청마다 모델을 새로 로드하면 VRAM/시간 낭비가 크기 때문에
#      전역 싱글톤 형태로 두는 것이 표준 패턴.
engine = BaseEngine()


# 4️⃣ Gradio ChatInterface 구성
#    fn으로 넘긴 함수는 Gradio가 (message: str, history: list) 형태로 호출한다.
#    engine.chat 이 이 시그니처를 만족해야 한다.
demo = gr.ChatInterface(
    fn=engine.chat,                           # 사용자 입력을 처리할 콜백
    title="🤖 기본 EXAONE 챗봇",               # 상단 제목
    description=engine.get_info_markdown(),   # 모델/설정 정보 (마크다운 렌더링)
    # 5️⃣ 예시 질문 버튼 — 클릭하면 입력창에 자동 입력된다.
    #    일반 대화 / 코딩 / 도메인(법률) / 일상 추천 등
    #    모델의 범용성을 보여줄 수 있는 다양한 카테고리로 구성.
    examples=[
        "안녕하세요, 자기소개 해주세요",
        "파이썬 리스트와 튜플의 차이는?",
        "임차인이 보증금을 돌려받지 못할 때 어떻게 해야 하나요?",
        "오늘 저녁 메뉴 추천해주세요",
    ],
)


# 6️⃣ 엔트리 포인트
#    - `python -m ui.chat_basic` 처럼 직접 실행할 때만 서버가 뜬다.
#    - 다른 모듈이 import 할 때는 launch()가 실행되지 않아
#      데모를 재조립/테스트하기 편하다.
if __name__ == "__main__":
    demo.launch()
    # 💡 참고 옵션:
    #   demo.launch(share=True)        → 임시 공개 URL 생성 (외부 공유용)
    #   demo.launch(server_name="0.0.0.0", server_port=7860)
    #                                  → 서버 배포 시 외부 접속 허용