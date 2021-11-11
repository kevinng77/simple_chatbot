import torch
import pathlib
import os
import argparse

root_path = pathlib.Path(__file__).parent.resolve()
intent_path = root_path / "intent"
bi_path = root_path / "bislot"

print(intent_path)
print(root_path)
raw_train = root_path.parent /"CrossWOZ/data/train.json"
raw_dev = root_path.parent/"CrossWOZ/data/val.json"

slot_file = intent_path / "data/slot.txt"

# intent
if not os.path.exists(intent_path/"data"):
    os.mkdir(intent_path/"data")
intent_train = intent_path / "data/train_intent.csv"
intent_dev = intent_path / "data/dev_intent.csv"
intent_weight = intent_path / "checkout/state_dict/intent_model.pth"


bi_train = bi_path / "data/train_bi.csv"
bi_dev = bi_path / "data/dev_bi.csv"
bi_weight = intent_path / "checkout/state_dict/bi_model.pth"




is_cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if is_cuda else torch.device('cpu')


def set_seed(seed, is_cuda):
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name",type=str,default="hfl/rbt3")
    parser.add_argument('--train_file', default='../data/intent_train.json')
    parser.add_argument('--test_file', default='../data/intent_test.json')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--epoch', default=3)
    parser.add_argument('--seed', default=2021)
    parser.add_argument('--max_seq_len', default=128)
    parser.add_argument('--lr', default=2e-5)
    args = parser.parse_args()
    return args