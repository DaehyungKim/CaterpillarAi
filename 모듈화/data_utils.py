# data_utils.py - 데이터 처리 모듈
import json
import numpy as np
from datasets import Dataset, Features, Value, ClassLabel, Sequence
from sklearn.model_selection import train_test_split

def load_intent_data():
    """의도 분류 데이터 로드 함수"""
    # Intent 라벨 로드
    with open('intent_label_list.json', 'r', encoding='utf-8') as f:
        intent_label_list = json.load(f)

    intent_label_to_id = {label: i for i, label in enumerate(intent_label_list)}
    intent_id_to_label = {i: label for label, i in intent_label_to_id.items()}

    # Intent 데이터 로드
    with open('intent_data.json', 'r', encoding='utf-8') as f:
        intent_data = json.load(f)

    return intent_data, intent_label_list, intent_label_to_id, intent_id_to_label

def load_ner_data():
    """개체명 인식 데이터 로드 함수"""
    with open('ner_data.json', 'r', encoding='utf-8') as f:
        loaded_ner_data = json.load(f)

    # 데이터 형식 변환
    ner_data = []
    for item in loaded_ner_data:
        entities_as_tuples = [tuple(entity_list) for entity_list in item.get("entities", [])]
        ner_data.append({"text": item.get("text", ""), "entities": entities_as_tuples})

    return ner_data

def prepare_intent_dataset(intent_data, intent_label_list):
    """의도 분류 데이터셋 준비 함수"""
    num_intent_labels = len(intent_label_list)

    intent_features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=num_intent_labels, names=intent_label_list)
    })

    intent_dataset = Dataset.from_list(intent_data, features=intent_features)
    train_test_datasets = intent_dataset.train_test_split(test_size=0.2, seed=42)

    return train_test_datasets["train"], train_test_datasets["test"]

def preprocess_intent_data(examples, tokenizer, max_len):
    """의도 분류 데이터 전처리 함수"""
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_len
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

def prepare_ner_dataset(ner_data, tokenizer, max_len):
    """NER 데이터셋 준비 및 전처리 함수"""
    # 엔티티 타입 추출
    entity_types = sorted(list(set(label for item in ner_data for _, _, label in item["entities"])))

    # BIO 태그 생성
    ner_labels = ["O"]  # Outside tag
    for entity_type in entity_types:
        ner_labels.extend([f"B-{entity_type}", f"I-{entity_type}"])

    ner_label2id = {label: i for i, label in enumerate(ner_labels)}
    ner_id2label = {i: label for label, i in ner_label2id.items()}

    # 데이터 전처리
    preprocessed_ner_data = []

    for example in ner_data:
        text = example["text"]
        entities = example["entities"]

        # 문자 단위 BIO 태깅
        char_labels = ["O"] * len(text)

        # 엔티티에 따라 BIO 태그 할당
        for start_char, end_char, entity_type in entities:
            # 범위 검사 및 조정
            if start_char < 0:
                start_char = 0
            if end_char > len(text):
                end_char = len(text)

            if start_char < end_char and start_char < len(text):
                for i in range(start_char, end_char):
                    if i == start_char:
                        char_labels[i] = f"B-{entity_type}"
                    else:
                        char_labels[i] = f"I-{entity_type}"

        # 토큰화 및 토큰-문자 정렬
        tokenized = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
        offset_mapping = tokenized['offset_mapping']

        # 토큰별 레이블 할당
        token_labels = []
        for i, (start, end) in enumerate(offset_mapping):
            # 특수 토큰 처리
            if start == end:
                token_label = -100  # ignore_index
            else:
                # 토큰 시작 위치의 문자 레이블 사용
                char_label = char_labels[start]
                token_label = ner_label2id[char_label]

                # 서브워드 토큰은 I- 태그로 변환
                if i > 0 and tokens[i].startswith("##"):
                    prev_label = token_labels[-1]
                    if prev_label != -100 and ner_id2label[prev_label].startswith("B-"):
                        entity_type = ner_id2label[prev_label][2:]
                        token_label = ner_label2id[f"I-{entity_type}"]

            token_labels.append(token_label)

        # 데이터 추가
        preprocessed_ner_data.append({
            "text": text,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": token_labels
        })

    # 데이터셋 생성
    ner_features = Features({
        'text': Value('string'),
        'input_ids': Sequence(Value('int32')),
        'attention_mask': Sequence(Value('int32')),
        'labels': Sequence(Value('int32'))
    })

    ner_dataset = Dataset.from_list(preprocessed_ner_data, features=ner_features)
    train_test_datasets = ner_dataset.train_test_split(test_size=0.2, seed=42)

    return train_test_datasets["train"], train_test_datasets["test"], ner_labels, ner_label2id, ner_id2label