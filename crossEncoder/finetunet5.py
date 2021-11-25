import argparse

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


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

    device = torch.device('cuda')
    torch.manual_seed(123)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    train_samples = []
    with open(args.triples_path, 'r', encoding="utf-8") as fIn:
        for num, line in enumerate(fIn):
            if num > 6.4e5 * args.epochs:
                break
            query, positive, negative = line.split("\t")
            train_samples.append((query, positive, 'true'))
            train_samples.append((query, negative, 'false'))

    def smart_batching_collate_text_only(batch):
        texts = []
        for example in batch:
            new_token = None
            token = tokenizer.tokenize(example['text'])
            if len(token) > 512:
                for idx, t in enumerate(token):
                    if t == '▁Document' and token[idx + 1] == ':':
                        if idx > (len(token) - idx):
                            len_reduce = len(token) - 511
                            new_token = token[:(idx - len_reduce)] + token[idx:]
                            break
                if new_token:
                    texts.append(''.join(new_token).replace('▁', ' '))
                else:
                    texts.append(example['text'])
            else:
                texts.append(example['text'])
        tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']
        for idx, input_ids in enumerate(tokenized['input_ids']):
            if len(input_ids) == 512 and input_ids[-1] == 1 and input_ids[-4] != 31484:
                tokenized['input_ids'][-4] = 31484
                tokenized['input_ids'][-3] = 17
                tokenized['input_ids'][-2] = 10
                tokenized['input_ids'][-1] = 1
        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)

        return tokenized

    dataset_train = MonoT5Dataset(train_samples)

    if args.save_every_n_steps:
        steps = args.save_every_n_steps
        strategy = 'steps'
    else:
        steps = 1
        strategy = 'epoch'

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,
        do_train=True,
        save_strategy='no',
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=5e-5,
        num_train_epochs=1,
        warmup_steps=1000,
        adafactor=True,
        seed=1,
        disable_tqdm=False,
        load_best_model_at_end=False,
        predict_with_generate=True,
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=smart_batching_collate_text_only,
    )

    trainer.train()

    trainer.save_model(args.output_model_path)
    trainer.save_state()


if __name__ == "__main__":
    main()
