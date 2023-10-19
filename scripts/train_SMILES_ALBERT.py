"""
Training Albert Language Model

author: mivler
10/18/2023
"""
import transformers
import datasets
import os
import torch
import argparse


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(prog="TrainMolecularAlbert", description="Script to train Albert language model on Molecular SMILES input")

    # Positional Args
    parser.add_argument("trainingfile", help="Training data file where each line is a unique SMILES string")
    parser.add_argument("validationfile", help="Validation data file where each line is a unique SMILES string")
    parser.add_argument("tokenizerdir", help="Directory where PreTrained huggingface tokenizer information is saved")
    # Optional Args
    parser.add_argument("-vs", "--vocab-size", type=int, default=2048, help="Size of tokenizer vocab")
    parser.add_argument("-bs", "--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("-o", "--out-dir", default="models", help="Directory to store trained Albert Model")
    
    args = parser.parse_args()

    # Check cuda availability and ensure cuda:0 is device used if so
    print(torch.cuda.is_available())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load in datsaet
    data_files = {"train": args.trainingfile, "valid": args.validationfile}
    dataset = datasets.load_dataset("data", data_files=data_files)

    # Gets max training string length
    max_length = int(max([len(string) for string in dataset["train"]["text"]]))

    # Loads Pretrained tokenizer
    tokenizer = transformers.AlbertTokenizerFast.from_pretrained(args.tokenizerdir)

    # Sets up function to tokenize input strings
    def encode_truncation(examples):
        """
        Function to tokenize input strings, padding to be equal length, and truncating if too long. Tokenized tensors are pytorch tensors.

        Arguments
        ---------
            examples - (datasets.Dataset) A batch of the dataset being tokenized

        Return
        ------
            (datasets.Dataset) Tokenized examples with with input ids and attention_mask
        """
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True, return_tensors="pt")

    # Take text from training and validation datasets and encodes them with tokenizer
    train_dataset = dataset["train"].map(encode_truncation, batched=True, batch_size=8)
    valid_dataset = dataset["valid"].map(encode_truncation, batched=True, batch_size=8)
    
    # Sets up Model Configuration and Albert model for Masked LM task
    model_config = transformers.AlbertConfig(vocab_size=args.vocab_size, max_position_embeddings=max_length)
    model = transformers.AlbertForMaskedLM(config=model_config)
    # Moves model to device (cuda:0 if GPUs available)
    model = model.to(device)

    # Initialize data_collator with default .15 proportion masking
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    # Sets up training arguments for trained
    training_args = transformers.TrainingArguments(output_dir=args.out_dir, evaluation_strategy="steps", overwrite_output_dir=True, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=2, per_device_eval_batch_size=args.batch_size, logging_steps=2000, save_steps=10000, load_best_model_at_end=True, save_total_limit=10)

    # Initilializes Trainer
    trainer = transformers.Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=valid_dataset)

    # Trains model with 
    trainer.train()
    
    # Save final model to output directory
    # This will be the best model because in training_args we set load_best_model_at_end to True
    trainer.save_model(os.path.join(args.out_dir, "best_model"))
    

if __name__ == "__main__":
    main()
