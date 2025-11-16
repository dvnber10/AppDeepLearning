# train_ocr.py
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import evaluate
import matplotlib.pyplot as plt

# Cargar dataset de escritura a mano (IAM Handwriting) - puedes cambiarlo por otro en español
dataset = load_dataset("huggingface/transformers", "iam_dataset_script")  # placeholder de ejemplo

# Preprocesador y modelo base
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Tokenización
def preprocess(batch):
    images = [x.convert("RGB") for x in batch["image"]]
    text = batch["text"]
    inputs = processor(images=images, text=text, padding="max_length", return_tensors="pt")
    inputs["labels"] = inputs.input_ids
    return inputs

dataset = dataset.map(preprocess, batched=True)

# Métrica
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

# Entrenamiento
training_args = Seq2SeqTrainingArguments(
    output_dir="./ocr_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    logging_dir="./logs_ocr",
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

trainer.train()

# Graficar loss
logs = trainer.state.log_history
train_loss = [x["loss"] for x in logs if "loss" in x]
eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]

plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, label="Eval Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("OCR Training Loss")
plt.legend()
plt.savefig("ocr_loss.png")

print("✅ OCR entrenado. Error CER final:", compute_metrics(trainer.predict(dataset["test"])))
