"""
ui/chat_evaluated.py — Stage 4: 통합 평가 비교 챗봇
====================================================
기본 EXAONE vs 파인튜닝 EXAONE 답변 동시 비교 + GPT 자동 평가

📌 이 파일의 역할
- **두 엔진을 동시에 로드**하고 같은 질문에 각각 응답시킴
- BERTScore + LLM Judge(GPT API)로 **자동 평가** → 승자 판정
- 단건 비교와 **배치 평가(테스트셋 전체 일괄)** 두 모드 모두 지원

🎯 앞선 파일들과의 차이
- chat_compare.py: 모델 1개 + 프롬프트 비교 → "프롬프트 엔지니어링 효과" 측정
- chat_evaluated.py: 모델 2개 + 동일 프롬프트 → "파인튜닝 효과" 측정

실행:
    cd exaone_chatbot
    python -m ui.chat_evaluated
"""
import gradio as gr
from core.base_engine import BaseEngine
from core.finetuned_engine import FinetunedEngine
from evaluation.metrics import evaluate_response
from evaluation.test_questions import get_all


# ═════════════════════════════════════════════════════════
# 1️⃣ 두 엔진 동시 로드
# ═════════════════════════════════════════════════════════
#  두 모델이 동시에 VRAM에 올라가므로 최소 ~1.5GB 추가 필요
#  (1.2B × 4bit × 2개 = 약 1.2GB + 오버헤드)
print("=" * 60)
print("🚀 통합 평가 챗봇 초기화")
print("=" * 60)

# 1-1. 기본 모델 로드 (항상 성공해야 함)
print("\n[1/2] 기본 EXAONE 로드...")
base_engine = BaseEngine()

# 1-2. 파인튜닝 모델 로드 — Stage 3 미실행 시 graceful degradation
#      try/except로 감싸서 **앱이 죽지 않도록** 함
#      → 파인튜닝 모델이 없어도 기본 모델은 여전히 써볼 수 있게
print("\n[2/2] 파인튜닝 EXAONE 로드...")
try:
    finetuned_engine = FinetunedEngine()
    FINETUNED_AVAILABLE = True
except FileNotFoundError as e:
    # FinetunedEngine 생성자의 친절한 에러 메시지를 그대로 표시
    print(f"⚠️ 파인튜닝 모델 로드 실패:\n{e}")
    finetuned_engine = None
    FINETUNED_AVAILABLE = False  # 플래그로 이후 분기 제어

# 1-3. 초기화 결과 요약
print("\n" + "=" * 60)
print(f"✅ 초기화 완료")
print(f"   🅰️ 기본:     {base_engine.label}")
print(f"   🅱️ 파인튜닝: {finetuned_engine.label if FINETUNED_AVAILABLE else '❌ 없음'}")
print("=" * 60 + "\n")


# ═════════════════════════════════════════════════════════
# 2️⃣ Helper 함수들
# ═════════════════════════════════════════════════════════

def find_reference(question):
    """테스트셋에서 참조 답변 찾기 (BERTScore 계산용)"""
    # 2-1. 완전 일치 우선 → 앞 20자 유연 매칭 → None
    #   Evaluator._find_reference와 동일한 로직
    for q in get_all():
        if q["question"] == question:
            return q["reference"]
        if q["question"][:20] == question[:20]:
            return q["reference"]
    return None


def format_eval_short(scores):
    """간단 점수 표시 (UI용 마크다운)"""
    # 2-2. 평가 결과가 없으면 안내만
    if not scores:
        return "_평가 결과 없음_"

    lines = []
    bs = scores.get("bertscore")
    j = scores.get("judge")

    # 2-3. BERTScore 표시 (임계값 기반 3단계 아이콘)
    #   ≥0.85 → 우수 ✅
    #   ≥0.70 → 양호 🟡
    #    <0.70 → 개선필요 🔴
    if bs:
        f1 = bs["F1"]
        icon = "✅" if f1 >= 0.85 else ("🟡" if f1 >= 0.7 else "🔴")
        lines.append(f"**BERTScore**: {f1:.3f} {icon}")

    # 2-4. LLM Judge 표시 (40점 만점)
    #   ≥32 (80%) → 우수 🟢
    #   ≥24 (60%) → 양호 🟡
    #    <24 → 개선필요 🔴
    if j:
        total = j["총점"]
        icon = "🟢" if total >= 32 else ("🟡" if total >= 24 else "🔴")
        lines.append(f"**LLM Judge**: {total}/40 {icon}")

        # 세부 항목별 점수 (법률 도메인 특화 루브릭)
        lines.append(
            f"_법률정확성:{j.get('법률_정확성',0)} · "
            f"법조항인용:{j.get('법조항_인용',0)} · "
            f"법적절차:{j.get('법적_절차',0)} · "
            f"상담품질:{j.get('법률_상담_품질',0)}_"
        )

        # Judge가 제공한 자연어 코멘트 (있을 때만)
        근거 = j.get("평가_근거", "")
        if 근거:
            lines.append(f"💬 _{근거}_")

    return "\n\n".join(lines) if lines else "_평가 비활성_"


def format_winner(scores_a, scores_b):
    """승자 판정 마크다운 생성"""
    # 2-5. 한쪽이라도 점수가 없으면 판정 스킵
    if not scores_a or not scores_b:
        return ""

    # 2-6. 안전한 점수 추출 (nested dict + 기본값 0)
    #   .get(...).get(...) 연쇄는 None이 껴있으면 AttributeError 유발
    #   따라서 먼저 dict 존재 여부를 and로 확인
    j_a = scores_a.get("judge", {}).get("총점", 0) if scores_a.get("judge") else 0
    j_b = scores_b.get("judge", {}).get("총점", 0) if scores_b.get("judge") else 0
    bs_a = scores_a.get("bertscore", {}).get("F1", 0) if scores_a.get("bertscore") else 0
    bs_b = scores_b.get("bertscore", {}).get("F1", 0) if scores_b.get("bertscore") else 0

    lines = ["### 🏆 종합 판정"]

    # 2-7. Judge 점수 기반 승자 결정 (주요 지표)
    if j_a or j_b:
        if j_a > j_b:
            lines.append(f"- **LLM Judge**: 🅰️ 기본 승 (+{j_a-j_b}점, {j_a} vs {j_b})")
        elif j_b > j_a:
            lines.append(f"- **LLM Judge**: 🅱️ 파인튜닝 승 (+{j_b-j_a}점, {j_b} vs {j_a})")
        else:
            lines.append(f"- **LLM Judge**: 🤝 무승부 ({j_a} = {j_b})")

    # 2-8. BERTScore 기반 승자 (부차 지표)
    #   0.01 미만 차이는 "거의 동일"로 간주 — 측정 오차 범위
    if bs_a or bs_b:
        if abs(bs_a - bs_b) < 0.01:
            lines.append(f"- **BERTScore**: 🤝 거의 동일 ({bs_a:.3f} ≈ {bs_b:.3f})")
        elif bs_a > bs_b:
            lines.append(f"- **BERTScore**: 🅰️ 기본 승 ({bs_a:.3f} vs {bs_b:.3f})")
        else:
            lines.append(f"- **BERTScore**: 🅱️ 파인튜닝 승 ({bs_a:.3f} vs {bs_b:.3f})")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════
# 3️⃣ 메인 비교 + 평가 함수
# ═════════════════════════════════════════════════════════
def compare_and_evaluate(question, use_bertscore, use_judge):
    """두 엔진 동시 실행 + 평가 + 승자 판정"""
    # 3-1. 빈 입력 방어 (5개 output 모두 반환해야 함)
    if not question.strip():
        return ("", "", "", "", "_질문을 입력하세요_")

    # 3-2. 콘솔 로그 — 시연 시 진행 상황 실시간 확인
    print(f"\n{'=' * 60}")
    print(f"🎯 비교 실행: {question[:50]}...")
    print(f"{'=' * 60}")

    # 3-3. 참조 답변 조회 (있으면 BERTScore 가능)
    reference = find_reference(question)
    if reference:
        print(f"✅ 참조 답변 있음 (BERTScore 활성)")
    else:
        print(f"⚠️ 참조 답변 없음 (BERTScore 불가)")

    # 3-4. 🅰️ 기본 모델 추론
    print("\n[1/2] 🅰️ 기본 EXAONE 응답 생성...")
    answer_base = base_engine.chat(question)
    print(f"   ✅ 완료 ({len(answer_base)}자)")

    # 3-5. 🅱️ 파인튜닝 모델 추론 (없으면 안내 메시지)
    if FINETUNED_AVAILABLE:
        print("\n[2/2] 🅱️ 파인튜닝 EXAONE 응답 생성...")
        answer_ft = finetuned_engine.chat(question)
        print(f"   ✅ 완료 ({len(answer_ft)}자)")
    else:
        # graceful degradation — 파인튜닝 없어도 UI는 동작
        answer_ft = (
            "⚠️ 파인튜닝 모델이 준비되지 않았습니다.\n\n"
            "필요 작업:\n"
            "1. notebooks/01_finetune.ipynb 실행\n"
            "2. notebooks/02_merge_model.ipynb 실행\n"
            "3. 이 앱 재시작"
        )

    # 3-6. 평가 수행 (BERTScore 즉시 / Judge는 API 호출)
    print("\n📊 GPT API 평가 진행 중...")

    print("   🅰️ 기본 모델 평가...")
    scores_base = evaluate_response(
        question=question,
        answer=answer_base,
        reference=reference,
        use_bertscore=use_bertscore,  # 체크박스 값
        use_judge=use_judge,
    )
    if scores_base.get("judge"):
        print(f"      Judge: {scores_base['judge']['총점']}/40")

    if FINETUNED_AVAILABLE:
        print("   🅱️ 파인튜닝 모델 평가...")
        scores_ft = evaluate_response(
            question=question,
            answer=answer_ft,
            reference=reference,
            use_bertscore=use_bertscore,
            use_judge=use_judge,
        )
        if scores_ft.get("judge"):
            print(f"      Judge: {scores_ft['judge']['총점']}/40")
    else:
        scores_ft = None

    # 3-7. UI 표시용 마크다운 포맷팅
    eval_base_md = format_eval_short(scores_base)
    eval_ft_md = format_eval_short(scores_ft) if scores_ft else "_파인튜닝 모델 없음_"
    winner_md = format_winner(scores_base, scores_ft) if scores_ft else ""

    print("\n✅ 비교 완료\n")

    # 3-8. 5개 output 반환 (answer_a, eval_a, answer_b, eval_b, winner)
    return (answer_base, eval_base_md, answer_ft, eval_ft_md, winner_md)


# ═════════════════════════════════════════════════════════
# 4️⃣ 배치 평가 (테스트셋 전체 일괄)
# ═════════════════════════════════════════════════════════
def run_batch_compare(use_bertscore, use_judge, progress=gr.Progress()):
    """테스트셋 10건 전체 비교
    
    progress=gr.Progress(): Gradio가 자동으로 진행률 바를 그려줌
    → 긴 작업(5~10분)에서 사용자 체감 시간 개선
    """
    test_set = get_all()
    progress(0, desc=f"{len(test_set)}개 질문 평가 시작...")

    results = []

    # 4-1. 각 질문마다 순차 비교
    for i, item in enumerate(test_set, 1):
        # 진행률 업데이트 (0.0 ~ 1.0)
        progress(
            i / len(test_set),
            desc=f"[{i}/{len(test_set)}] {item['id']}: {item['question'][:30]}...",
        )

        # 4-2. 두 모델 응답 생성 (history 없이 독립 추론 → 공정성)
        ans_base = base_engine.chat(item["question"])
        ans_ft = finetuned_engine.chat(item["question"]) if FINETUNED_AVAILABLE else ""

        # 4-3. 두 응답 평가
        scores_base = evaluate_response(
            item["question"], ans_base, item["reference"],
            use_bertscore=use_bertscore, use_judge=use_judge,
        )
        scores_ft = evaluate_response(
            item["question"], ans_ft, item["reference"],
            use_bertscore=use_bertscore, use_judge=use_judge,
        ) if FINETUNED_AVAILABLE else None

        results.append({
            "id": item["id"],
            "question": item["question"],
            "domain": item["domain"],
            "scores_base": scores_base,
            "scores_ft": scores_ft,
        })

    # 4-4. 집계 → 마크다운 테이블
    return format_batch_summary(results)


def format_batch_summary(results):
    """배치 결과 마크다운 (표 형태)"""
    # 5️⃣ 평균 계산 헬퍼 (nested closure)
    #    클로저로 results를 캡처 → 중복 코드 제거
    def avg_judge(key):
        vals = [
            r[key]["judge"]["총점"]
            for r in results
            if r[key] and r[key].get("judge")
        ]
        return sum(vals) / len(vals) if vals else 0.0

    def avg_bertscore(key):
        vals = [
            r[key]["bertscore"]["F1"]
            for r in results
            if r[key] and r[key].get("bertscore")
        ]
        return sum(vals) / len(vals) if vals else 0.0

    # 5-1. 두 모델 전체 평균
    base_j = avg_judge("scores_base")
    ft_j = avg_judge("scores_ft")
    base_bs = avg_bertscore("scores_base")
    ft_bs = avg_bertscore("scores_ft")

    # 5-2. 질문별 승패 카운트 (Judge 점수 기준)
    base_wins, ft_wins, ties = 0, 0, 0
    for r in results:
        # 안전한 점수 추출
        b = r["scores_base"].get("judge", {}).get("총점", 0) if r["scores_base"] and r["scores_base"].get("judge") else 0
        f = r["scores_ft"].get("judge", {}).get("총점", 0) if r["scores_ft"] and r["scores_ft"].get("judge") else 0
        if b > f:
            base_wins += 1
        elif f > b:
            ft_wins += 1
        else:
            ties += 1

    # 5-3. 전체 요약 헤더 + 평균 표
    #   `+.1f`, `+.3f`: 양수일 때도 부호 표시 (차이를 명확히)
    md = f"""### 📊 배치 평가 결과 ({len(results)}건)

#### 평균 점수
| 지표 | 🅰️ 기본 EXAONE | 🅱️ 파인튜닝 | 차이 |
|---|---|---|---|
| **LLM Judge** (40점) | **{base_j:.1f}** | **{ft_j:.1f}** | {ft_j - base_j:+.1f} |
| **BERTScore F1** | {base_bs:.3f} | {ft_bs:.3f} | {ft_bs - base_bs:+.3f} |

#### 질문별 승패 (Judge 기준)
- 🅰️ 기본 승:    **{base_wins}** 건
- 🅱️ 파인튜닝 승: **{ft_wins}** 건
- 🤝 무승부:      **{ties}** 건

#### 질문별 상세
| # | 도메인 | 🅰️ | 🅱️ | 승자 |
|---|---|---|---|---|
"""

    # 5-4. 각 질문 행 추가
    for r in results:
        b_j = r["scores_base"]["judge"]["총점"] if r["scores_base"] and r["scores_base"].get("judge") else "-"
        f_j = r["scores_ft"]["judge"]["총점"] if r["scores_ft"] and r["scores_ft"].get("judge") else "-"

        # 정수 타입 검사 후 비교 (- 문자열일 수 있으므로)
        if isinstance(b_j, int) and isinstance(f_j, int):
            if b_j > f_j:
                winner = "🅰️"
            elif f_j > b_j:
                winner = "🅱️"
            else:
                winner = "🤝"
        else:
            winner = "-"

        md += f"| {r['id']} | {r['domain']} | {b_j} | {f_j} | {winner} |\n"

    md += f"\n💾 상세 로그는 `evaluation/logs/`에 저장됩니다."
    return md


# ═════════════════════════════════════════════════════════
# 6️⃣ Gradio UI (버전 호환, 최소 파라미터만 사용)
# ═════════════════════════════════════════════════════════
with gr.Blocks(title="EXAONE 통합 평가 비교") as demo:
    # 6-1. 헤더 + 모델 정보 표
    gr.Markdown("# 🎯 EXAONE 통합 평가 비교 챗봇")
    gr.Markdown(f"""
**기본 EXAONE** vs **파인튜닝 EXAONE** 동시 비교 + **GPT API 자동 평가**

| 구분 | 모델 |
|---|---|
| 🅰️ 기본 | {base_engine.label} |
| 🅱️ 파인튜닝 | {finetuned_engine.label if FINETUNED_AVAILABLE else '❌ 미로드 (01/02 노트북 먼저 실행)'} |
""")

    # 6-2. 평가 옵션 토글
    #      BERTScore/Judge를 사용자가 On/Off 할 수 있게
    #      → API 비용 없이 빠른 체감만 할 때는 둘 다 False
    with gr.Row():
        use_bs = gr.Checkbox(label="BERTScore 사용 (참조 답변 필요)", value=True)
        use_judge = gr.Checkbox(label="LLM Judge 사용 (OpenAI API 비용 발생)", value=True)

    gr.Markdown("---")

    # 6-3. 질문 입력
    question = gr.Textbox(
        label="❓ 테스트 질문",
        placeholder="법률 관련 질문을 입력하세요...",
        lines=2,
    )

    submit_btn = gr.Button("▶️ 두 모델 동시 실행 + GPT 평가", variant="primary")

    # 6-4. 승자 판정 표시 영역 (버튼 바로 아래 → 눈에 잘 띔)
    winner_display = gr.Markdown()

    # 6-5. 답변 비교 (좌: 기본, 우: 파인튜닝)
    gr.Markdown("## 📋 답변 비교")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🅰️ 기본 EXAONE (순정)")
            answer_base = gr.Textbox(label="답변", lines=15)
            eval_base = gr.Markdown(label="평가")

        with gr.Column():
            gr.Markdown("### 🅱️ 파인튜닝 EXAONE")
            answer_ft = gr.Textbox(label="답변", lines=15)
            eval_ft = gr.Markdown(label="평가")

    # 6-6. 예시 질문
    gr.Examples(
        examples=[
            ["임차인이 보증금을 돌려받지 못할 때 어떻게 해야 하나요?"],
            ["근로계약서 없이 일한 경우 임금 체불 시 대처법은?"],
            ["교통사고 후 보험사와 합의 시 주의사항은?"],
            ["상속 포기와 한정승인의 차이는?"],
            ["이웃 간 소음 분쟁 법적 대응 방법은?"],
            ["이혼 시 재산분할 비율은 어떻게 결정되나요?"],
            ["주택연금 가입자가 사망하면 어떻게 되나요?"],
        ],
        inputs=question,
    )

    gr.Markdown("---")

    # 6-7. 배치 평가 섹션 (페이지 하단)
    #   secondary 버튼 → 주 기능(단건 비교)과 시각적 구분
    gr.Markdown("## 🔬 전체 테스트셋 배치 평가")
    gr.Markdown(
        "_테스트셋 10개 질문으로 **두 모델 일괄 비교**. "
        "시간 걸림 (5~10분), API 비용 발생._"
    )
    batch_btn = gr.Button("▶️ 전체 테스트셋 일괄 비교", variant="secondary")
    batch_result = gr.Markdown()

    # 7️⃣ 이벤트 바인딩
    #   단건 비교 — 버튼/엔터 두 경로 모두 지원
    submit_btn.click(
        compare_and_evaluate,
        inputs=[question, use_bs, use_judge],
        outputs=[answer_base, eval_base, answer_ft, eval_ft, winner_display],
    )
    question.submit(
        compare_and_evaluate,
        inputs=[question, use_bs, use_judge],
        outputs=[answer_base, eval_base, answer_ft, eval_ft, winner_display],
    )
    # 배치 비교 — 버튼만 (엔터 트리거 없음)
    batch_btn.click(
        run_batch_compare,
        inputs=[use_bs, use_judge],
        outputs=batch_result,
    )


# 8️⃣ 엔트리 포인트
if __name__ == "__main__":
    demo.launch()