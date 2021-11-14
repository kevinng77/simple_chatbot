import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import re


class Tokenizer:
    def __init__(self, args, ner_label2id, ner_id2label):
        self.bert_tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
        self.pad_token = self.bert_tokenizer.pad_token

        self.max_seq_len = args.max_seq_len

        self.vocab = ner_label2id
        self.id2vocab = ner_id2label
        self.target_pad_token_id = 99

        self.vocab[self.pad_token] = self.target_pad_token_id
        self.id2vocab[self.target_pad_token_id] = self.pad_token

    def edit_len(self, text_ids, is_target):
        """
        padding ids for data batch
        is_target bool: is padding for target token ids?
        """
        if is_target:
            padding = self.target_pad_token_id
            if len(text_ids) > self.max_seq_len:
                return text_ids[:self.max_seq_len]
            else:
                return text_ids + [padding] * (self.max_seq_len - len(text_ids))

        else:
            padding = self.bert_tokenizer.pad_token_id
            if len(text_ids) > self.max_seq_len:
                return text_ids[:self.max_seq_len]
            else:
                return text_ids + [padding] * (self.max_seq_len - len(text_ids))

    def tokens_to_ids(self, text, is_target=False):
        """
        text: list[str] list of words in a sentence.
        """
        if is_target:
            sequence = [self.vocab[x] for x in text]
        else:
            sequence = self.bert_tokenizer.convert_tokens_to_ids(text)

        if len(sequence) == 0:
            sequence = [0]
        return self.edit_len(sequence, is_target)

    def text_to_ids(self, text, is_target=False):
        sequence = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        return self.edit_len(sequence, is_target)

    def text_to_tokens(self, text, train=True, aspect=None):
        tokens = self.bert_tokenizer.tokenize(text)
        return tokens

    def ids_to_tokens(self, ids, is_target=False):
        if is_target:
            sequence = [self.id2vocab[x] for x in ids if x != self.target_pad_token_id]
        else:
            sequence = [self.bert_tokenizer._convert_id_to_token(x)
                        for x in ids if x != self.bert_tokenizer.pad_token_id]
        return sequence


class NerDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, "r") as fp:
            lines = fp.readlines()
        dataset = []
        for i in range(0, len(lines), 2):
            text = lines[i]
            gold = lines[i + 1]
            inputs = tokenizer.bert_tokenizer(text.strip(),
                                              max_length=tokenizer.max_seq_len,
                                              return_tensors='pt',
                                              padding="max_length"
                                              )

            pred_tokens = ["O"] + gold.strip().split(",") + ["O"]
            pred_ids = tokenizer.tokens_to_ids(pred_tokens, is_target=True)
            # pred_ids = [tokenizer.target_pad_token_id] + pred_ids + [tokenizer.target_pad_token_id]
            # att_mask = [1 if x != tokenizer.target_pad_token_id else 0 for x in pred_ids]
            # data = {
            #     "text_ids": text,
            #     "att_mask": torch.tensor(att_mask,dtype=torch.long),
            #     "pred_ids": torch.tensor(pred_ids,dtype=torch.long)
            # }
            inputs = {k: v[0] for k, v in inputs.items()}
            inputs['labels'] = torch.tensor(pred_ids, dtype=torch.long)
            dataset.append(inputs)
            assert inputs['input_ids'].shape[0] == inputs['labels'].shape[0], \
                f"{inputs['input_ids'].shape[0]} != {inputs['labels'].shape[0]}"
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def find_index(query, item):
    frag = query.split(item, 1)
    if len(frag) == 1:
        return -1
    return len(frag[0])


def generate_ner_data(file_path, save_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        query_list = []
        for task_id, item in data.items():
            for i, turn in enumerate(item['messages']):
                if turn['role'] == 'usr':
                    query = turn['content']
                    query = re.sub("\s", " ", query)
                    ner_n = 0
                    tmp = ['O'] * len(query)
                    for j, each in enumerate(turn['dialog_act']):
                        if each[0] == "Inform" and each[3] and '酒店设施' not in each[2]:
                            entity = each[3]
                            idx = find_index(query, entity)
                            if idx > -1:
                                ner_n += 1
                                tmp[idx] = 'B_' + each[1] + '-' + each[2]
                                tmp[idx + 1:idx + len(entity)] = ['I_' + each[1] + '-' + each[2]] * (len(entity) - 1)
                    if ner_n > 0:
                        tmp = ','.join(tmp)
                        query_list.append([query, tmp])

    with open(save_path, "w") as fp:
        for query, label in query_list:
            fp.write(f"{query}\n")
            fp.write(f"{label}\n")


if __name__ == '__main__':
    import sys

    sys.path.append("../..")
    import config

    args = config.get_args()
    generate_ner_data(config.raw_train, config.ner_train)
    generate_ner_data(config.raw_dev, config.ner_dev)
