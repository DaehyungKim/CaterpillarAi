# main.py - 메인 모듈
import os
import sys
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('nlu_training.log')
    ]
)
logger = logging.getLogger(__name__)

# 모듈 임포트
from 모듈화.config import *
from 모듈화.data_utils import load_intent_data, load_ner_data, prepare_intent_dataset, prepare_ner_dataset
from 모듈화.intent_model import train_intent_model, load_intent_model
from 모듈화.ner_model import train_ner_model, load_ner_model
from 모듈화.predict import predict

def train_models():
    """모든 모델 훈련 함수"""
    logger.info("=" * 50)
    logger.info("도서 검색 NLU 모델 훈련 시작")
    logger.info("=" * 50)

    # 1. 데이터 로드
    intent_data, intent_label_list, intent_label_to_id, intent_id_to_label = load_intent_data()
    ner_data = load_ner_data()
    logger.info(f"Intent 레이블 ({len(intent_label_list)}개) 로드 완료")
    logger.info(f"NER 데이터 ({len(ner_data)}개) 로드 완료")

    # 2. 의도 분류 데이터셋 준비
    intent_train_dataset, intent_eval_dataset = prepare_intent_dataset(intent_data, intent_label_list)
    logger.info(f"Intent 훈련 데이터: {len(intent_train_dataset)}개, 평가 데이터: {len(intent_eval_dataset)}개")

    # 3. 의도 분류 모델 훈련
    intent_model, intent_tokenizer = train_intent_model(
        intent_train_dataset,
        intent_eval_dataset,
        MODEL_NAME,
        intent_id_to_label,
        intent_label_to_id,
        EPOCHS,
        BATCH_SIZE,
        "./results/intent",
        INTENT_MODEL_DIR
    )

    # 4. NER 데이터셋 준비
    ner_train_dataset, ner_eval_dataset, ner_labels, ner_label2id, ner_id2label = prepare_ner_dataset(
        ner_data,
        intent_tokenizer,  # 같은 토크나이저 재사용
        MAX_LEN
    )
    logger.info(f"NER 훈련 데이터: {len(ner_train_dataset)}개, 평가 데이터: {len(ner_eval_dataset)}개")
    logger.info(f"NER 레이블 ({len(ner_labels)}개) 생성 완료")

    # 5. NER 모델 훈련
    ner_model, ner_tokenizer = train_ner_model(
        ner_train_dataset,
        ner_eval_dataset,
        MODEL_NAME,
        ner_labels,
        ner_id2label,
        ner_label2id,
        EPOCHS,
        BATCH_SIZE,
        MAX_LEN,
        "./results/ner",
        NER_MODEL_DIR
    )

    logger.info("=" * 50)
    logger.info("도서 검색 NLU 모델 훈련 완료")
    logger.info("=" * 50)

    return intent_model, intent_tokenizer, intent_id2label, ner_model, ner_tokenizer, ner_id2label

def load_models():
    """저장된 모델 로드 함수"""
    logger.info("저장된 모델 로드 중...")

    # Intent 모델 로드
    try:
        intent_model, intent_tokenizer, intent_id2label = load_intent_model(INTENT_MODEL_DIR, INTENT_LABEL_PATH)
        logger.info(f"Intent 모델 로드 완료 (레이블 수: {len(intent_id2label)})")
    except Exception as e:
        logger.error(f"Intent 모델 로드 실패: {e}")
        intent_model, intent_tokenizer, intent_id2label = None, None, None

    # NER 모델 로드
    try:
        ner_model, ner_tokenizer, ner_id2label = load_ner_model(NER_MODEL_DIR, NER_LABEL_PATH)
        logger.info(f"NER 모델 로드 완료 (레이블 수: {len(ner_id2label)})")
    except Exception as e:
        logger.error(f"NER 모델 로드 실패: {e}")
        ner_model, ner_tokenizer, ner_id2label = None, None, None

    return intent_model, intent_tokenizer, intent_id2label, ner_model, ner_tokenizer, ner_id2label

def test_prediction(text, intent_model, intent_tokenizer, intent_id2label, ner_model, ner_tokenizer, ner_id2label):
    """예측 테스트 함수"""
    logger.info(f"테스트 입력: '{text}'")
    result = predict(
        text,
        intent_model,
        intent_tokenizer,
        intent_id2label,
        ner_model,
        ner_tokenizer,
        ner_id2label,
        MAX_LEN
    )

    logger.info(f"예측 의도: {result['intent']} (신뢰도: {result['confidence']:.4f})")
    logger.info(f"추출된 엔티티: {result['entities']}")
    logger.info(f"문자별 태그: {result['char_tags']}")

    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="도서 검색 NLU 시스템")
    parser.add_argument("--train", action="store_true", help="모델 훈련 실행")
    parser.add_argument("--test", type=str, help="예측 테스트 입력 문장")
    args = parser.parse_args()

    if args.train:
        # 모델 훈련
        intent_model, intent_tokenizer, intent_id2label, ner_model, ner_tokenizer, ner_id2label = train_models()
    else:
        # 저장된 모델 로드
        intent_model, intent_tokenizer, intent_id2label, ner_model, ner_tokenizer, ner_id2label = load_models()

    if args.test:
        # 예측 테스트
        test_prediction(args.test, intent_model, intent_tokenizer, intent_id2label, ner_model, ner_tokenizer, ner_id2label)