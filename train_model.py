# fine_tune_kaggle_resume_ner.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

# 1️⃣ Load Kaggle dataset
print("Loading Kaggle resume dataset...")
dataset = load_dataset("yashpwr/resume-ner-training-data")

print(f"Available splits: {dataset.keys()}")
train_dataset = dataset["train"]
# Create a small validation split
train_test_split = train_dataset.train_test_split(test_size=0.05)
dataset = {
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
}

print(f"Training samples: {len(dataset['train'])}, Validation samples: {len(dataset['validation'])}")

# 2️⃣ Load pre-trained model and tokenizer
model_name = "yashpwr/resume-ner-bert-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# 3️⃣ Define labels (use model's labels)
label_list = list(model.config.id2label.values())
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

# 4️⃣ Tokenization + dummy label alignment
def tokenize_and_align(batch):
    # Extract text from messages (chat format)
    # Each sample has messages list with dicts containing 'content' and 'role'
    # We'll concatenate the user messages (typically the resume text)
    texts = []
    for messages in batch["messages"]:
        # Find the user message containing the resume
        user_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
                break
        texts.append(user_content if user_content else "")
    
    tokenized_inputs = tokenizer(
        texts,  # Extracted text strings
        truncation=True,
        max_length=512,
        padding="max_length",
        is_split_into_words=False,  # raw text
    )

    # Assign all tokens to 'O' (non-entity) as dummy labels
    labels = []
    for input_ids in tokenized_inputs["input_ids"]:
        labels.append([label2id["O"]] * len(input_ids))

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing dataset...")
tokenized_dataset = {}
for split in ["train", "validation"]:
    tokenized_dataset[split] = dataset[split].map(
        tokenize_and_align, batched=True, remove_columns=dataset[split].column_names
    )

# 5️⃣ Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# 6️⃣ Training arguments
training_args = TrainingArguments(
    output_dir="./models/kaggle-resume-ner",
    eval_strategy="epoch",  # Updated from evaluation_strategy
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,  # Use 1 for testing; increase later
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=20,
    report_to="none",
    fp16=torch.cuda.is_available()
)

# 7️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 8️⃣ Train
print("Starting fine-tuning with dummy labels...")
trainer.train()

# 9️⃣ Save
trainer.save_model("./models/kaggle-resume-ner")
tokenizer.save_pretrained("./models/kaggle-resume-ner")

print("\n✅ Fine-tuned model saved! You can now use it with a pipeline like this:")
print("from transformers import pipeline")
print("ner = pipeline('ner', model='./models/kaggle-resume-ner', aggregation_strategy='simple')")
 