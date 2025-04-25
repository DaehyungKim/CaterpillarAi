# intent_model.py - 의도 분류 모델 모듈
import json
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score

def compute_intent_metrics(eval_pred):
    """의도 분류 평가 지표 계산 함수"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train_intent_model(train_dataset, eval_dataset, model_name, intent_id_to_label,
                       intent_label_to_id, epochs, batch_size, output_dir, save_dir):
    """의도 분류 모델 훈련 함수"""
    print("\n" + "="*50)
    print("Intent 분류 모델 훈련 시작")
    print("="*50)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 데이터 전처리
    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=64
        )

    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=['text']
    )

    tokenized_eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=['text']
    )

    # 데이터 콜레이터 설정
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(intent_id_to_label),
        id2label=intent_id_to_label,
        label2id=intent_label_to_id
    )

    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        eval_steps=100,
        eval_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    # Trainer 정의
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_intent_metrics,
    )

    # 모델 훈련
    trainer.train()
    eval_result = trainer.evaluate()
    print(f"Intent 모델 평가 결과: {eval_result}")

    # 모델 및 레이블 저장
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)

    label_info = {
        "id2label": intent_id_to_label,
        "label2id": intent_label_to_id
    }

    label_path = os.path.join(save_dir, "intent_labels.json")
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)

    print(f"Intent 모델 및 레이블 정보 저장 완료: {save_dir}")

    return model, tokenizer

def load_intent_model(model_dir, label_path):
    """저장된 의도 분류 모델 로드 함수"""
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 레이블 정보 로드
    with open(label_path, 'r', encoding='utf-8') as f:
        label_info = json.load(f)

    id2label = {int(k): v for k, v in label_info["id2label"].items()}
    label2id = label_info["label2id"]

    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )

    return model, tokenizer, id2label

def predict_intent(model, tokenizer, id2label, text, max_len=64):
    """의도 예측 함수"""
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    )

    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    if torch.cuda.is_available():
        model = model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        predicted_intent = id2label[predicted_class_id]

    return {
        "intent": predicted_intent,
        "confidence": probabilities.cpu().numpy().flatten().tolist()
    }