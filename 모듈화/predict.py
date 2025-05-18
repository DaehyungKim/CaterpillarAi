# predict.py - 예측 모듈
from 모듈화.intent_model import predict_intent
from 모듈화.ner_model import predict_entities

def predict(text, intent_model, intent_tokenizer, intent_id2label,
            ner_model, ner_tokenizer, ner_id2label, max_len=64):
    """통합 예측 함수"""
    if not intent_model or not ner_model:
        return {
            "text": text,
            "intent": "오류: 모델 로드 실패",
            "confidence": 0.0,
            "entities": [],
            "error": "모델 로드 실패"
        }

    try:
        # 의도 예측
        intent_result = predict_intent(intent_model, intent_tokenizer, intent_id2label, text, max_len)

        # 개체명 예측
        ner_result = predict_entities(ner_model, ner_tokenizer, ner_id2label, text, max_len)

        # 문자별 태그 시각화 추가
        char_tags = ner_result.get("char_tags", ["_"] * len(text))

        # 결과 통합
        return {
            "text": text,
            "intent": intent_result["intent"],
            "confidence": max(intent_result["confidence"]) if intent_result.get("confidence") else 0.0,
            "entities": ner_result["entities"],
            "ner_tokens": ner_result.get("tokens", []),
            "ner_token_probabilities": ner_result.get("token_probabilities", []),
            "char_tags": "".join(char_tags)
        }

    except Exception as e:
        import traceback
        print(f"예측 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "text": text,
            "intent": "예측 오류",
            "confidence": 0.0,
            "entities": [],
            "ner_tokens": [],
            "ner_token_probabilities": [],
            "error": str(e)
        }