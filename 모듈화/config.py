# config.py - 설정 모듈
import os

# 기본 설정
MODEL_NAME = "klue/bert-base"
MAX_LEN = 64
EPOCHS = 5
BATCH_SIZE = 4

# 모델 저장 경로 설정
INTENT_MODEL_DIR = "./models/intent"
NER_MODEL_DIR = "./models/ner"
INTENT_LABEL_PATH = os.path.join(INTENT_MODEL_DIR, "intent_labels.json")
NER_LABEL_PATH = os.path.join(NER_MODEL_DIR, "ner_labels.json")