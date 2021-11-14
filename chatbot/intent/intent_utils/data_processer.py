from torch.utils.data import Dataset
import pandas as pd
import json
import random
import torch


def generate_data(slots, file_path, save_path):
    with open(file_path) as f:
        data = json.load(f)
        query_list = []
        for task_id, item in data.items():
            for i, turn in enumerate(item['messages']):
                if turn['role'] == 'usr':
                    query = turn['content']
                    labels = set()
                    m = 0
                    for k, act in enumerate(turn['dialog_act']):
                        if act[0] in ["General","Request"]:
                            labels.add(act[1]+'-'+act[2])
                        elif act[0] == "Select":
                            labels.add(act[1]+"-名称")
                    for label in labels:
                        query_list.append([query, label, 1])
                        m += 1
                    if m < 15:  # 15 用于调整正负样本比例
                        tmp = [x for x in slots if x not in labels]
                        neg_samples = random.sample(tmp, 15-m)
                        for neg_label in neg_samples:
                            query_list.append([query, neg_label, 0])

        dataset = pd.DataFrame(query_list, columns=['query', 'slot', 'label'])
        dataset.to_csv(save_path)


class IntentDataset(Dataset):
    def __init__(self, filename, tokenizer, args):
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.seqs, self.seq_masks, self.seq_segments, self.labels \
            = self.get_input(filename)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[
            idx], self.labels[idx]

    def get_input(self, filename):
        df = pd.read_csv(filename, index_col=0)
        df['slot'] = df['slot'].apply(lambda x: '有在问' + x + '吗')
        labels = df['label'].astype('int64').values
        print(df.head(2))
        # tokenize the sentences
        tokens_seq_1 = list(
            map(self.tokenizer.tokenize, df['query'].values))
        tokens_seq_2 = list(
            map(self.tokenizer.tokenize, df['slot'].values))
        result = list(map(self.create_seq, tokens_seq_1, tokens_seq_2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(
            torch.long), torch.Tensor(seq_masks).type(
            torch.long), torch.Tensor(seq_segments).type(
            torch.long), torch.Tensor(labels).type(torch.long)

    def create_seq(self, tokens_seq_1, tokens_seq_2):
        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) +
                             2) + [1] * (len(tokens_seq_2) + 1)
        seq = self.tokenizer.convert_tokens_to_ids(seq)
        padding = [0] * (self.max_seq_len - len(seq))
        seq_mask = [1] * len(seq) + padding
        seq_segment = seq_segment + padding
        seq = seq + padding
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment


if __name__ == '__main__':
    import sys
    sys.path.append("../..")
    import config

    slots = config.slots
    generate_data(slots, config.raw_train, config.intent_train)
    generate_data(slots, config.raw_dev, config.intent_dev)
