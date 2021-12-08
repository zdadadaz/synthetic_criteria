import argparse

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    get_constant_schedule,
    AdamW
)
from transformers import T5Config


class MonoT5Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f'Query: {sample[0]} Document: {sample[1]} Relevant:'
        return {
            'text': text,
            'labels': sample[2],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='t5-base', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--triples_path", default=None, type=str, required=True,
                        help="Triples.tsv path")
    parser.add_argument("--triples_path_eval", default=None, type=str, required=True,
                        help="eval Triples.tsv path")
    parser.add_argument("--output_model_path", default=None, type=str, required=True,
                        help="Path for trained model and checkpoints.")
    parser.add_argument("--save_every_n_steps", default=0, type=int, required=False,
                        help="Save every N steps. (recommended 10000)")
    parser.add_argument("--logging_steps", default=100, type=int, required=False,
                        help="Logging steps parameter.")
    parser.add_argument("--per_device_train_batch_size", default=8, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int, required=False,
                        help="Gradient accumulation parameter.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, required=False,
                        help="Learning rate parameter.")
    parser.add_argument("--epochs", default=10, type=int, required=False,
                        help="Number of epochs to train")
    parser.add_argument("--gradient_checkpointing", default='False', choices=('True', 'False'),
                        help="train large model")
    parser.add_argument("--model_parallel", type=int, default=0, help="model parallel")
    parser.add_argument("--max_steps", default=0, type=int, required=False,
                        help="Number of steps to train")

    device = torch.device('cuda')
    torch.manual_seed(123)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    def read_tripple(path_to_tripple):
        train_samples = []
        with open(path_to_tripple, 'r', encoding="utf-8") as fIn:
            for num, line in enumerate(fIn):
                if num > 6.4e5 * args.epochs:
                    break
                query, pos_doc, neg_doc = line.split("\t")
                train_samples.append((query, pos_doc, 'true'))
                train_samples.append((query, neg_doc, 'false'))
        return train_samples

    def smart_batching_collate_text_only(batch):
        texts = [example['text'] for example in batch]
        tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']
        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)

        return tokenized

    train_samples = read_tripple(args.triples_path)
    eval_samples = read_tripple(args.triples_path_eval)

    dataset_train = MonoT5Dataset(train_samples)
    dataset_eval = MonoT5Dataset(eval_samples)

    if args.save_every_n_steps:
        steps = args.save_every_n_steps
        strategy = 'steps'
    else:
        steps = 1
        strategy = 'epoch'

    if args.model_parallel:
        model.parallelize()

    model.config.use_cache = False if args.gradient_checkpointing == 'True' else True
    train_args = Seq2SeqTrainingArguments(
        gradient_checkpointing=True if args.gradient_checkpointing == 'True' else False,
        output_dir=args.output_model_path,
        do_train=True,
        evaluation_strategy='steps',
        save_strategy='steps',
        eval_steps=200,
        save_steps=200,
        num_train_epochs=1,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0,
        weight_decay=0,
        adafactor=False,
        seed=1,
        disable_tqdm=False,
        load_best_model_at_end=False,
        predict_with_generate=True,
        dataloader_pin_memory=False
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_constant_schedule(optimizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
        data_collator=smart_batching_collate_text_only,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()

    trainer.save_model(args.output_model_path)
    trainer.save_state()


if __name__ == "__main__":
    main()
