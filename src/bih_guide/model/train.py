import os
from pathlib import Path

import torch
import yaml

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import  LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT_DIR / "configs" / "training.yaml"
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "sft" / "train.jsonl"
VALIDATION_PATH = ROOT_DIR / "data" / "processed" / "sft" / "validation.jsonl"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "adapter_smoke"


if __name__ == "__main__":
    print("====================================")
    print("ENVIRONMENT")
    print("====================================")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU nije dostupan. QLoRA smoke test zahtijeva NVIDIA GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n====================================")
    print("LOADING CONFIG")
    print("====================================")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    print(config)

    torch.manual_seed(config["seed"])

    print("\n====================================")
    print("LOADING DATASET")
    print("====================================")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_PATH),
            "validation": str(VALIDATION_PATH),
        },
    )

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    print(f"Number train examples: {len(train_dataset)}")
    print(f"Number validation examples: {len(validation_dataset)}")

    print("\n====================================")
    print("LOADING TOKENIZER")
    print("====================================")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n====================================")
    print("FIRST DATASET EXAMPLE")
    print("====================================")

    first_example = train_dataset[0]
    print("example_id:")
    print(first_example["example_id"])

    print("\ntype:")
    print(first_example["type"])

    print("\ndestination_ids:")
    print(first_example["destination_ids"])

    print("\nsource_ids:")
    print(first_example["source_ids"])

    print("\nreviewed:")
    print(first_example["reviewed"])

    print("\nmessages:")
    for message in first_example["messages"]:
        print(f"\nROLE: {message['role']}")
        print(f"CONTENT: {message['content']}")

    print("\n====================================")
    print("FORMATTED CHAT EXAMPLE")
    print("====================================")

    formatted_example = tokenizer.apply_chat_template(
        first_example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    print(formatted_example)

    print("\n====================================")
    print("FORMATTING DATASET")
    print("====================================")


    def format_example(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }


    train_dataset = train_dataset.map(
        format_example,
        remove_columns=train_dataset.column_names,
    )

    validation_dataset = validation_dataset.map(
        format_example,
        remove_columns=validation_dataset.column_names,
    )

    print("Dataset formatting finished.")
    print("\nFormatted training example:")
    print(train_dataset[0]["text"])

    print("\n====================================")
    print("PREPARING 4-BIT QUANTIZATION")
    print("====================================")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n====================================")
    print("LOADING QWEN MODEL")
    print("====================================")

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    print("Model loaded successfully.")

    print("\n====================================")
    print("PREPARING MODEL FOR K-BIT TRAINING")
    print("====================================")

    model = prepare_model_for_kbit_training(model)

    print("\n====================================")
    print("CREATING LORA CONFIG")
    print("====================================")

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(lora_config)

    print("\n====================================")
    print("CREATING TRAINING ARGUMENTS")
    print("====================================")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_steps=2,
        per_device_train_batch_size=config[
            "per_device_train_batch_size"
        ],
        gradient_accumulation_steps=config[
            "gradient_accumulation_steps"
        ],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        fp16=True,
        report_to="none",
        seed=config["seed"],
    )

    print("\n====================================")
    print("CREATING SFT TRAINER")
    print("====================================")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("\n====================================")
    print("STARTING QLORA SMOKE TEST")
    print("====================================")

    print("max_steps = 2")
    print("batch_size = 1")
    print("gradient_accumulation_steps = 8")
    print("LoRA r = 16")
    print("LoRA alpha = 32")
    print("====================================\n")

    result = trainer.train()

    print("\n====================================")
    print("TRAINING FINISHED")
    print("====================================")

    print("\nTraining result:")
    print(result)

    if result.training_loss is not None:
        print(f"\nTraining loss: {result.training_loss}")

        if torch.isnan(torch.tensor(result.training_loss)):
            raise RuntimeError( "Training loss is NaN.")

    print("\n====================================")
    print("SAVING LORA ADAPTER")
    print("====================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"Adapter saved to:")
    print(OUTPUT_DIR)

    del trainer
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n====================================")
    print("RELOADING BASE MODEL")
    print("====================================")

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model.config.use_cache = True
    print("Base model loaded.")

    print("\n====================================")
    print("LOADING LORA ADAPTER")
    print("====================================")

    model = PeftModel.from_pretrained(
        base_model,
        str(OUTPUT_DIR),
    )
    model.eval()
    print("Adapter loaded successfully.")

    print("\n====================================")
    print("GENERATION TEST")
    print("====================================")

    messages = [
        {
            "role": "system",
            "content": (
                "Ti si pouzdan turistički vodič za Bosnu i Hercegovinu. "
                "Odgovaraj na srpskom jeziku, ijekavicom i latinicom. "
                "Koristi samo dostavljene provjerene informacije. "
                "Ne izmišljaj cijene, radna vremena, redove vožnje, "
                "datume događaja ni druge promjenjive podatke."
            ),
        },
        {
            "role": "user",
            "content": "Koji je glavni grad Bosne i Hercegovine?",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\nQuestion:")
    print(messages[-1]["content"])

    print("\nModel's answer:")
    print(answer)

    print("\n====================================")
    print("QLORA SMOKE TEST SUCCESSFULLY FINISHED")
    print("====================================")
