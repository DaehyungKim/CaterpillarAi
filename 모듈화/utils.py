# utils.py - 유틸리티 모듈
import torch
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

def free_gpu_memory():
    """GPU 메모리 정리 함수"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU 메모리 정리 완료")

def visualize_intent_confusion_matrix(y_true, y_pred, labels):
    """의도 분류 혼동 행렬 시각화 함수"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.title('의도 분류 혼동 행렬')
    plt.tight_layout()
    plt.savefig('intent_confusion_matrix.png')
    plt.close()
    logger.info("의도 분류 혼동 행렬 이미지 저장 완료")

def visualize_ner_results(text, entities, char_tags):
    """개체명 인식 결과 시각화 함수"""
    # 텍스트에 엔티티 표시
    highlighted_text = text

    # 엔티티별 색상 매핑
    colors = {
        "genre": "\033[91m",  # 빨강
        "author": "\033[92m",  # 녹색
        "title": "\033[94m",   # 파랑
        "publisher": "\033[95m"  # 보라색
    }

    # 복원 코드
    end_color = "\033[0m"

    # 위치를 기준으로 역순 정렬 (겹침 방지)
    sorted_entities = sorted(entities, key=lambda x: text.find(x["entity"]), reverse=True)

    # 엔티티에 색상 적용
    for entity in sorted_entities:
        entity_text = entity["entity"]
        entity_type = entity["label"]

        if entity_type in colors:
            color_code = colors[entity_type]
            start_pos = text.find(entity_text)

            if start_pos != -1:
                highlighted_text = (
                    highlighted_text[:start_pos] +
                    color_code + entity_text + end_color +
                    highlighted_text[start_pos + len(entity_text):]
                )

    # 결과 출력
    logger.info(f"원본 텍스트: {text}")
    logger.info(f"시각화된 텍스트: {highlighted_text}")
    logger.info(f"문자별 태그: {''.join(char_tags)}")
    logger.info("엔티티 목록:")

    for entity in sorted_entities:
        entity_type = entity["label"]
        entity_text = entity["entity"]
        logger.info(f"  - {entity_type}: {entity_text}")

    return highlighted_text