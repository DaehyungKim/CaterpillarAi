# ner_model.py - 개체명 인식 모델 모듈
import json
import os
import gc
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import f1_score, classification_report

def compute_ner_metrics(p):
    """NER 평가 지표 계산 함수"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 실제 토큰의 예측값과 레이블만 추출
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        true_pred = []
        true_label = []

        for p, l in zip(prediction, label):
            if l != -100:  # -100은 무시
                true_pred.append(id2label[p])
                true_label.append(id2label[l])

        true_predictions.append(true_pred)
        true_labels.append(true_label)

    # seqeval의 f1_score 계산
    try:
        f1 = f1_score(true_labels, true_predictions)
        report = classification_report(true_labels, true_predictions, digits=4)
        print("\nNER Classification Report:\n", report)

        return {"f1": f1}
    except Exception as e:
        print(f"Error calculating NER metrics: {e}")
        return {"f1": 0.0}

def train_ner_model(train_dataset, eval_dataset, model_name, ner_labels, ner_id2label,
                   ner_label2id, epochs, batch_size, max_len, output_dir, save_dir):
    """NER 모델 훈련 함수"""
    print("\n" + "="*50)
    print("NER 모델 훈련 시작")
    print("="*50)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 데이터 콜레이터 설정
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_len,
        pad_to_multiple_of=8
    )

    # 모델 로드
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(ner_labels),
        id2label=ner_id2label,
        label2id=ner_label2id
    )

    # 컴퓨트 메트릭스 함수 생성
    def compute_metrics(p):
        return compute_ner_metrics(p)

    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=2,
        greater_is_better=True,
        metric_for_best_model="f1",
        weight_decay=0.01,
        report_to="none",
    )

    # Trainer 정의
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 모델 훈련
    trainer.train()
    eval_result = trainer.evaluate()
    print(f"NER 모델 평가 결과: {eval_result}")

    # 모델 및 레이블 저장
    os.makedirs(save_dir, exist_ok=True)
    gc.collect()
    model.cpu()
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)

    label_info = {
        "id2label": ner_id2label,
        "label2id": ner_label2id
    }

    label_path = os.path.join(save_dir, "ner_labels.json")
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)

    print(f"NER 모델 및 레이블 정보 저장 완료: {save_dir}")

    return model, tokenizer

def load_ner_model(model_dir, label_path):
    """저장된 NER 모델 로드 함수"""
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # 레이블 정보 로드
    with open(label_path, 'r', encoding='utf-8') as f:
        label_info = json.load(f)

    id2label = {int(k): v for k, v in label_info["id2label"].items()}
    label2id = label_info["label2id"]

    # 모델 로드
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )

    return model, tokenizer, id2label

def predict_entities(model, tokenizer, id2label, text, max_len=64):
    """개체명 예측 함수"""
    if not model or not tokenizer or not id2label:
        return {"tokens": [], "tags": [], "entities": [], "token_probabilities": []}

    model.eval()

    # 토큰화 시 오프셋 매핑 반환
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True,
                     max_length=max_len, return_offsets_mapping=True)

    # 오프셋 매핑 별도 저장 및 제거
    offset_mapping = inputs.pop("offset_mapping").cpu().numpy()[0]

    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    # GPU 사용 가능 시 이동
    if torch.cuda.is_available():
        model = model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=2)
        predictions = torch.argmax(logits, dim=2)

    # 결과 추출
    input_ids = inputs["input_ids"][0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    predicted_tag_ids = predictions[0].cpu().numpy()
    all_probabilities = probabilities[0].cpu().numpy()

    # 특수 토큰 제외 및 결과 처리
    content_tokens = []
    content_tags = []
    token_probabilities_list = []
    content_offsets = []

    for i, (token, tag_id, input_id, (start, end)) in enumerate(zip(tokens, predicted_tag_ids, input_ids, offset_mapping)):
        if input_id == tokenizer.pad_token_id:
            break

        if token not in tokenizer.all_special_tokens:
            content_tokens.append(token)
            tag = id2label.get(tag_id, "O")
            content_tags.append(tag)
            content_offsets.append((start, end))
            token_probabilities_list.append(all_probabilities[i].tolist())

    # 엔티티 추출
    entities_found = []
    current_entity_tokens = []
    current_entity_starts = []
    current_entity_ends = []
    current_entity_label = None

    for i, (token, tag, (start, end)) in enumerate(zip(content_tokens, content_tags, content_offsets)):
        if tag.startswith("B-"):
            # 이전 엔티티 처리
            if current_entity_tokens:
                entity_text = text[current_entity_starts[0]:current_entity_ends[-1]]
                entities_found.append({
                    "entity": entity_text,
                    "label": current_entity_label
                })

            # 새 엔티티 시작
            current_entity_tokens = [token]
            current_entity_starts = [start]
            current_entity_ends = [end]
            current_entity_label = tag[2:]

        elif tag.startswith("I-") and current_entity_label == tag[2:]:
            # 기존 엔티티 계속
            current_entity_tokens.append(token)
            current_entity_starts.append(start)
            current_entity_ends.append(end)

        else:
            # 엔티티 종료
            if current_entity_tokens:
                entity_text = text[current_entity_starts[0]:current_entity_ends[-1]]
                entities_found.append({
                    "entity": entity_text,
                    "label": current_entity_label
                })
            current_entity_tokens = []
            current_entity_starts = []
            current_entity_ends = []
            current_entity_label = None

    # 마지막 엔티티 처리
    if current_entity_tokens:
        entity_text = text[current_entity_starts[0]:current_entity_ends[-1]]
        entities_found.append({
            "entity": entity_text,
            "label": current_entity_label
        })

    # 문자 단위 시각화를 위한 태그 매핑
    char_tags = ["_"] * len(text)

    for entity in entities_found:
        entity_text = entity["entity"]
        entity_type = entity["label"]

        # 원본 텍스트에서 엔티티 위치 찾기
        start_pos = text.find(entity_text)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)

            # 위치에 태그 표시
            char_tags[start_pos] = "B"
            for i in range(start_pos + 1, end_pos):
                char_tags[i] = "I"

    return {
        "tokens": content_tokens,
        "tags": content_tags,
        "entities": entities_found,
        "token_probabilities": token_probabilities_list,
        "char_tags": char_tags
    }