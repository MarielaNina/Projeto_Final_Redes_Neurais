from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import load_dataset
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

from preprocessing import flatten_data, prepare_train_features
from postprocessing import postprocess_qa_predictions

def custom_data_collator(features):
    """
    Envuelve default_data_collator y elimina la clave 'labels' si existe,
    para evitar pasar 'labels' a BertForQuestionAnswering (que no la acepta).
    """
    batch = default_data_collator(features)

    # DEBUG: puedes activar la siguiente línea para ver las claves del batch la primera vez
    # print("Batch keys before cleanup:", list(batch.keys()))

    if 'labels' in batch:
        # Si 'labels' existe lo removemos. Asumimos que `prepare_train_features`
        # ya creó 'start_positions' y 'end_positions' (lo habitual).
        batch.pop('labels')

    # DEBUG: ver claves después (opcional)
    # print("Batch keys after cleanup:", list(batch.keys()))
    return batch


def run():

    # ------------ DATASET ------------
    train_file = './data/flat_squad-train-v1.1.json' \
        if Path('./data/flat_squad-train-v1.1.json').exists() else flatten_data('./data/squad-train-v1.1.json')

    validation_file = './data/flat_squad-dev-v1.1.json' \
        if Path('./data/flat_squad-dev-v1.1.json').exists() else flatten_data('./data/squad-dev-v1.1.json')

    qa_dataset = load_dataset(
        'json', 
        data_files={'train': train_file, 'validation': validation_file}, 
        field='data'
    )

    model_type = "base"
    model_name = f"neuralmind/bert-{model_type}-portuguese-cased"

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    tokenized_datasets = qa_dataset.map(
        prepare_train_features,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 512, "stride": 128, "padding_right": True},
        batched=True,
        remove_columns=qa_dataset["train"].column_names
    )

    # ------------ MODELO BASE ------------
    model = BertForQuestionAnswering.from_pretrained(model_name)

    # ------------ CONFIGURACIÓN LORA ------------
    lora_config = LoraConfig(
        #task_type=TaskType.FEATURE_EXTRACTION,
        task_type=TaskType.QUESTION_ANS,
        r=16,                # Ajustado según tu deseo de probar r=16
        lora_alpha=32,       # alpha proporcional a r
        lora_dropout=0.1,    # Dropout
        bias="none",
        target_modules=["query", "key", "value"]   # Incluimos key también
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Para verificar LoRA

    # ------------ MÉTRICAS ------------
    metric = evaluate.load("squad")

    def compute_metrics(p):
        final_predictions = postprocess_qa_predictions(
            qa_dataset["validation"],
            tokenized_datasets["validation"],
            p.predictions
        )

        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in final_predictions.items()
        ]

        references = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in qa_dataset["validation"]
        ]

        return metric.compute(predictions=formatted_predictions, references=references)

    # ------------ TRAINING ARGS ------------
    training_args = TrainingArguments(
        output_dir="./results/lora_bertimbau_qa_r16",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=custom_data_collator,   # <-- usamos nuestro collator aquí
        compute_metrics=compute_metrics
    )

    print("Entrenando modelo LoRA (V2 ajustado)...")
    trainer.train()
    print("Evaluación final:")
    print(trainer.evaluate())


if __name__ == "__main__":
    run()
