"""
evaluation/metrics.py — 평가 지표
- BERTScore: 참조 답변과의 의미 유사도
- LLM-as-Judge: GPT-4o-mini가 4가지 기준 × 10점 채점

📌 이 파일의 역할
- "답변 품질"을 숫자로 측정하는 **두 가지 보완적 지표** 제공
- Evaluator/chat_evaluated가 이 모듈을 호출해 자동 평가 수행
- 외부 의존(bert_score, openai)은 함수 내부에서 lazy import → 전체 import 비용 최소화

🎯 두 지표의 역할 분담
    BERTScore  — 의미적 유사도 (참조 답변과 얼마나 비슷한가?)
                 빠름, 결정적, 로컬 실행, 무료
    LLM Judge  — 품질 세부 항목 채점 (법률 상담으로서 훌륭한가?)
                 느림, 비결정적, API 호출, 비용 발생
                 but: 문체·정확성·구조까지 종합 판단
"""
import json
from typing import Optional  # Python 3.9 이하 호환을 위한 Optional[X] = Union[X, None]


# ═══════════════════════════════════════════════════════
# 1️⃣ BERTScore — 임베딩 기반 유사도
# ═══════════════════════════════════════════════════════
def compute_bertscore(hypothesis: str, reference: str, lang: str = "ko") -> dict:
    """
    BERTScore 계산.
    Returns: {"P": ..., "R": ..., "F1": ...}
    
    📖 원리 요약
    - hypothesis와 reference를 각각 BERT로 임베딩
    - 토큰 단위 코사인 유사도 매트릭스 생성
    - P(Precision): hypothesis 각 토큰이 reference의 어떤 토큰과 가장 유사한가?
    - R(Recall):    reference 각 토큰이 hypothesis의 어떤 토큰과 가장 유사한가?
    - F1 = 2PR/(P+R) — 이 값이 "의미적 유사도"의 최종 지표
    
    📊 F1 해석 기준 (한국어 기준)
    - 0.85+ : 매우 유사 (거의 동일한 의미)
    - 0.70~0.84 : 유사 (핵심 내용 일치)
    - 0.70 미만: 의미 차이 큼
    """
    # 1-1. Lazy import — bert_score는 torch + HF 모델을 끌어와 무거움
    #   이 함수를 호출할 때만 로드하면 전체 앱 시작 속도 보호
    from bert_score import score as bertscore_fn
    
    # 1-2. 리스트로 감싸서 전달 (배치 API지만 여기선 1건씩)
    #   내부에서 한국어 BERT(예: klue/bert-base) 자동 다운로드 후 사용
    #   verbose=False: 진행률 출력 억제 (UI에서 거슬리므로)
    P, R, F1 = bertscore_fn(
        cands=[hypothesis],    # candidate(후보) = 모델 생성 답변
        refs=[reference],       # reference(참조) = 정답 답변
        lang=lang,              # "ko" → 한국어 BERT 자동 선택
        verbose=False,
    )
    
    # 1-3. tensor → float 변환 후 dict 반환
    #   [0] 슬라이스 이유: 입력이 1건이라 결과도 길이 1 텐서
    return {
        "P":  float(P[0]),
        "R":  float(R[0]),
        "F1": float(F1[0]),  # ← 주로 이 값만 본다
    }


# ═══════════════════════════════════════════════════════
# 2️⃣ LLM-as-Judge — GPT-4o-mini가 법률 답변 채점
# ═══════════════════════════════════════════════════════

# 2-1. 전역 변수로 OpenAI 클라이언트를 캐싱 (싱글톤 패턴)
#   매번 OpenAI() 를 호출하면 로그인/세션 설정 오버헤드 발생
_openai_client = None


def _get_client():
    """OpenAI 클라이언트 싱글톤"""
    # 2-2. 전역 변수 수정을 위해 global 선언 필수
    global _openai_client
    
    # 2-3. 처음 호출될 때만 초기화 — Lazy init
    if _openai_client is None:
        # 함수 내부 import — openai가 설치 안된 환경에서도 BERTScore는 쓸 수 있게
        from openai import OpenAI
        from dotenv import load_dotenv
        
        # .env 파일에서 OPENAI_API_KEY 자동 로드
        # OpenAI() 는 환경변수에서 키를 자동으로 읽음
        load_dotenv()
        _openai_client = OpenAI()
    
    return _openai_client


def compute_llm_judge(question: str, answer: str, 
                      reference: Optional[str] = None) -> Optional[dict]:
    """
    GPT-4o-mini로 법률 상담 품질 4가지 기준 채점 (각 1~10점, 총 40점).
    
    Returns:
        {"법률_정확성": N, "법조항_인용": N, "법적_절차": N, "법률_상담_품질": N, 
         "총점": N, "평가_근거": "..."}
        or None (실패 시)
    
    🎯 왜 GPT-4o-mini?
    - gpt-4o 대비 비용 ~15배 저렴 (60/240 → 15/60 $/1M token)
    - 채점 정도의 단순 판단은 mini로 충분 (창의성보다 판별력이 중요)
    - 초당 수십 건 호출 가능 → 배치 평가 실용적
    """
    # 3-1. 클라이언트 싱글톤 획득
    client = _get_client()
    
    # 3-2. 참조 답변이 있으면 프롬프트에 포함 (있을 때만)
    #   [:500] 으로 제한 — 토큰 절약 + 핵심만 전달
    ref_part = f"\n[참고 법률 상담 답변]\n{reference[:500]}" if reference else ""
    
    # 3-3. 평가 프롬프트 구성 — 매우 엄격한 루브릭
    #   핵심 설계 포인트:
    #   1) "엄격하게 채점하세요" 명시 → 기본 모델의 관대한 성향 억제
    #   2) ⚠️ 중요 원칙에서 "일반 상식은 법률 상담이 아님" 을 반복 강조
    #   3) 각 항목에 10/5/1 점 기준 예시 → 점수 분포 일관성 확보
    #   4) 마지막에 다시 한 번 "생활 상식은 가치 낮음" 재강조 (샌드위치 기법)
    #   5) JSON만 출력 강제 → 파싱 에러 최소화
    prompt = f"""당신은 대한민국 법률 전문가이자 엄격한 법률 상담 평가자입니다.
이 답변이 **법률 상담**으로서 얼마나 훌륭한지 엄격하게 채점하세요.

⚠️ 중요 원칙:
- 일반적인 생활 상식이나 응급조치 안내는 법률 상담이 아닙니다.
- 법적 권리·의무, 법조항, 법적 절차를 다루는 답변에 높은 점수를 주세요.
- "보험사에 연락하세요", "119 신고하세요" 같은 일반 상식은 법률 상담 가치가 낮습니다.
- "민법 제OOO조에 따라", "손해배상 청구권", "시효" 같은 법률 전문 내용이 중요합니다.

[질문]
{question}

[답변]
{answer[:800]}
{ref_part}

[채점 기준 - 각 1~10점]

1. **법률_정확성** (1~10): 법적 지식과 원칙이 정확한가?
   - 10: 법조항·판례·법리가 정확하고 오류 없음
   - 5: 일부 법률 용어 있으나 부정확하거나 부족
   - 1: 법적 내용 없음 (일반 상식만 나열)

2. **법조항_인용** (1~10): 관련 법조항을 구체적으로 인용하는가?
   - 10: "민법 제750조", "주택임대차보호법 제3조" 등 정확한 조항 인용
   - 5: "관련 법에 따라" 수준의 모호한 언급
   - 1: 법조항 언급 전혀 없음

3. **법적_절차** (1~10): 법적 권리 행사 절차를 구체적으로 안내하는가?
   - 10: 내용증명→소송→강제집행 등 법적 절차 명확히 제시
   - 5: 일부 법적 절차 언급하나 불완전
   - 1: 법적 절차 없이 일반 대응만 안내

4. **법률_상담_품질** (1~10): 변호사가 할 만한 법률 상담 수준인가?
   - 10: 법률 전문가가 답한 것처럼 권리·의무·구제방안 명확
   - 5: 부분적으로 법률 관점, 부분적으로 일반 안내
   - 1: 일반인이 해도 될 상식적 답변

⚠️ 주의: "경찰 신고", "병원 치료", "보험사 연락" 같은 **생활 상식**은 
법률 상담 관점에서 가치가 낮습니다. 법률 관점이 없으면 높은 점수 주지 마세요.

JSON만 출력:
{{"법률_정확성": 0, "법조항_인용": 0, "법적_절차": 0, "법률_상담_품질": 0, "총점": 0, "평가_근거": "왜 이 점수인지 2~3문장으로"}}"""
    
    # 3-4. API 호출 — try/except로 감싸 네트워크/쿼터 실패를 graceful 처리
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",                                   # 비용 효율 모델
            messages=[{"role": "user", "content": prompt}],        # 시스템 메시지 생략 — 단일 프롬프트
            max_tokens=400,                                         # 답변 길이 상한 (토큰 비용 제한)
            temperature=0,                                          # 결정적 출력 — 같은 답변엔 같은 점수
        )
        
        # 3-5. 응답 텍스트 추출 후 정리
        text = resp.choices[0].message.content.strip()
        
        # 3-6. 코드블록 제거 — GPT가 ```json ... ``` 로 감싸는 경우 방어
        #   split("```")[1]: 첫 번째 백틱 쌍 사이의 내용 추출
        #   .replace("json", ""): 언어 힌트 제거
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        # 3-7. JSON 파싱 → dict
        result = json.loads(text)
        
        # 3-8. 총점 재계산 (GPT 실수 방지)
        #   ⚠️ GPT가 종종 개별 항목과 총점이 안 맞는 JSON을 반환
        #     예: 8+7+9+6=30 인데 "총점": 32 로 응답
        #   → 파이썬에서 직접 합산해 덮어쓰기 = 안전장치
        result["총점"] = sum([
            result.get("법률_정확성", 0),   # .get()로 키 누락 방어
            result.get("법조항_인용", 0),
            result.get("법적_절차", 0),
            result.get("법률_상담_품질", 0),
        ])
        return result
    
    except Exception as e:
        # 3-9. 포괄적 예외 처리 (특정 에러 타입을 세분화하지 않음)
        #   가능한 에러들:
        #   - openai.APIError: API 쿼터 초과, 네트워크 문제
        #   - json.JSONDecodeError: GPT가 JSON 형식을 어긴 경우
        #   - KeyError: 키 누락
        #   모두 "평가 실패 → None 반환"으로 수렴 (호출자는 None 분기만 처리)
        print(f"⚠️  LLM Judge 실패: {e}")
        return None


# ═══════════════════════════════════════════════════════
# 4️⃣ 통합 평가 진입점
# ═══════════════════════════════════════════════════════
def evaluate_response(question: str, answer: str,
                      reference: Optional[str] = None,
                      use_bertscore: bool = True,
                      use_judge: bool = True) -> dict:
    """
    Returns:
        {"bertscore": {...} or None, "judge": {...} or None}
    
    🎯 설계 의도
    - 두 지표 중 어느 하나라도 실패해도 다른 지표는 정상 작동
    - 호출자(UI, Evaluator)는 반환 dict에서 key 존재 여부만 체크하면 됨
    """
    # 4-1. 초기화 — 실패/비활성 시 None으로 남도록
    result = {"bertscore": None, "judge": None}
    
    # 4-2. BERTScore 계산 조건: 플래그 ON + 참조 답변 존재
    #   참조 없이는 BERTScore 계산 불가 (유사도 비교 대상이 없음)
    if use_bertscore and reference:
        try:
            result["bertscore"] = compute_bertscore(answer, reference)
        except Exception as e:
            # bert_score 모델 로드 실패 등 대비
            print(f"⚠️  BERTScore 실패: {e}")
    
    # 4-3. LLM Judge는 참조 없어도 가능 (있으면 더 정확)
    if use_judge:
        result["judge"] = compute_llm_judge(question, answer, reference)
    
    return result


# ═══════════════════════════════════════════════════════
# 5️⃣ UI 표시용 마크다운 포맷터
# ═══════════════════════════════════════════════════════
def format_scores_markdown(scores: dict) -> str:
    """평가 결과를 마크다운으로 변환 (UI 표시용)"""
    lines = ["### 📊 평가 결과"]
    
    # 5-1. BERTScore 섹션
    bs = scores.get("bertscore")
    if bs:
        f1 = bs["F1"]
        # 3단계 임계값 아이콘
        #   ≥0.85: 매우 유사 ✅
        #   ≥0.70: 양호 🟡
        #    <0.70: 개선필요 🔴
        icon = "✅" if f1 >= 0.85 else ("🟡" if f1 >= 0.7 else "🔴")
        lines.append(f"\n**BERTScore F1**: {f1:.3f} {icon}")
        # Precision/Recall도 함께 표시 — F1만 보면 놓치는 정보 보완
        lines.append(f"- Precision: {bs['P']:.3f}, Recall: {bs['R']:.3f}")
    else:
        # None/비활성 시 이탤릭 안내 (_..._ → 이탤릭)
        lines.append("\n**BERTScore**: *참조 답변 없음 또는 비활성*")
    
    # 5-2. LLM Judge 섹션
    j = scores.get("judge")
    if j:
        total = j["총점"]
        # Judge 3단계 임계값 (40점 만점)
        #   ≥32 (80%): 우수 🟢
        #   ≥24 (60%): 양호 🟡
        #    <24: 개선필요 🔴
        icon = "🟢" if total >= 32 else ("🟡" if total >= 24 else "🔴")
        lines.append(f"\n**LLM Judge**: {total} / 40 {icon}")
        
        # 세부 항목 점수를 들여쓰기로 표시 (- prefix → 불릿)
        lines.append(f"- 법률_정확성:     {j.get('법률_정확성', 0)}")
        lines.append(f"- 법조항_인용:     {j.get('법조항_인용', 0)}")
        lines.append(f"- 법적_절차:       {j.get('법적_절차', 0)}")
        lines.append(f"- 법률_상담_품질:  {j.get('법률_상담_품질', 0)}")
        
        # Judge가 제공한 자연어 코멘트 (있을 때만)
        근거 = j.get("평가_근거", "")
        if 근거:
            # 💬 이모지 + 이탤릭 → 시각적으로 구분되는 "AI의 코멘트" 느낌
            lines.append(f"\n💬 _{근거}_")
    else:
        lines.append("\n**LLM Judge**: *실패 또는 비활성*")
    
    return "\n".join(lines)


# 6️⃣ 단독 실행 sanity check
#    `python -m evaluation.metrics` 로 평가 파이프라인이 제대로 도는지 확인
if __name__ == "__main__":
    # 테스트 — 법률 전문 답변에 대한 기대 점수: BERTScore 높음 + Judge 고득점
    question = "임차인이 보증금을 돌려받지 못할 때 어떻게 해야 하나요?"
    answer = "주택임대차보호법 제3조의2에 따라 내용증명 후 임차권등기명령을 신청할 수 있습니다."
    reference = "주택임대차보호법 제3조의2에 따라 임차인은 보증금 반환을 요구하고 소송을 제기할 수 있습니다."
    
    print("🔍 평가 테스트...")
    result = evaluate_response(question, answer, reference)
    # 포맷터 출력 — UI에서 보일 모양을 미리보기
    print(format_scores_markdown(result))