import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import readfile as rf
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-large-msmarco')
model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-large-msmarco')
model.to(device)

path_to_file = '../../../data/TRECCT2021/topics2021.xml'
topics = rf.read_topics_ct21(path_to_file)
sentence_number = 10
out = {}
for idx, topic in topics.items():
    input_ids = tokenizer.encode(topic, return_tensors='pt').to(device)
    out[idx] = []
    for _ in range(4):
        outputs = model.generate(
            input_ids=input_ids,
            max_length=512,
            do_sample=True,
            top_k=64,
            num_return_sequences=sentence_number)

        for i in range(sentence_number):
            if tokenizer.decode(outputs[i], skip_special_tokens=True) not in out[idx]:
                out[idx].append(tokenizer.decode(outputs[i], skip_special_tokens=True))
            # print(f'sample {i + 1}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')

json_object = json.dumps(out, indent=2)
with open('data/doc2query.json', "w") as outfile:
    outfile.write(json_object)