import torch
import pathlib
import os
import argparse

root_path = pathlib.Path(__file__).parent.resolve()
intent_path = root_path / "intent"
bi_path = root_path / "bislot"
ner_path = root_path / "ner"

print(intent_path)
print(root_path)
raw_train = root_path.parent /"CrossWOZ/data/train.json"
raw_dev = root_path.parent/"CrossWOZ/data/val.json"


slot_file = intent_path / "data/slot.txt"

# intent
intent_train = intent_path / "data/train_intent.csv"
intent_dev = intent_path / "data/dev_intent.csv"
intent_weight = intent_path / "checkout/state_dict/intent_model.pth"

# bi
bi_train = bi_path / "data/train_bi.csv"
bi_dev = bi_path / "data/dev_bi.csv"
bi_weight = bi_path / "checkout/state_dict/bi_model.pth"

# ner
ner_train = ner_path / "data/train_ner.csv"
ner_dev = ner_path / "data/dev_ner.csv"
ner_weight = ner_path / "checkout/state_dict/ner_model.pth"


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
    parser.add_argument('--batch_size', default=64,type=int)
    parser.add_argument('--epoch', default=10,type=int)
    parser.add_argument('--max_seq_len', default=128,type=int)
    parser.add_argument('--lr', default=5e-5,type=float)
    parser.add_argument("--num_classes",default=71,type=int,help="num classes for ner")

    parser.add_argument("--dropout",default=0.1,type=float)
    parser.add_argument("--step", type=int, default=100,
                        help="checkout for each _ number of training step, default 100")
    parser.add_argument("--downstream", type=str, default="linear",
                        help="linear, crf, lstm, san or lstm-crf")

    # downstream attention heads
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Default 12. number of attention heads for additional SAN")
    parser.add_argument("--num_layers", type=int, default=1, help="Default 1. number of LSTM layers")

    #  training param
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="default 1e-4")
    parser.add_argument("--loss", type=str, default="CE", help="'CE' or 'focal")
    parser.add_argument("--gamma", type=float, default=2.0, help="gamma for focal loss")
    parser.add_argument("--alpha", type=float, default=0.75, help="alpha for focal loss")
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument("--load_model", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=7, help="default 7")
    parser.add_argument("--metrics", type=str, default="f1", help="f1.")
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--max_grad_norm", type=float, default=2.0, help="limit max grad norm")
    parser.add_argument("--clip_large_grad", default=False, action='store_true',
                        help="clip large gradient before update optimize")

    # optimizer param
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--adam_amsgrad", default=False, action='store_true')

    # data augument test
    parser.add_argument("--augument", default=False, action='store_true')
    parser.add_argument("--contrastive", default=False, action='store_true')
    parser.add_argument("--rdrop", default=False, action='store_true')
    parser.add_argument("--rdrop_alpha", type=float, default=0.01)
    parser.add_argument("--temp", type=float, default=0.05)  # referred to simcse

    args = parser.parse_args()
    return args


def label_process(filename):
    with open(filename) as f:
        slots = f.read().split('\n')

    ner_label2id, ner_id2label = {'O': 0}, {0: 'O'}
    bi_label = []
    ner_n = 0
    for label in slots:
        if label.split('-')[1] != '酒店设施':
            ner_label2id['B_' + label] = 2 * ner_n + 1
            ner_label2id['I_' + label] = 2 * ner_n + 2
            ner_id2label[2 * ner_n + 1] = 'B_' + label
            ner_id2label[2 * ner_n + 2] = 'I_' + label
            ner_n += 1
        else:
            bi_label.append(label)

    slots += ['greet-none', 'thank-none', 'bye-none']
    slots2id = {slot: i for i, slot in enumerate(slots)}
    id2slots = {i: slot for i, slot in enumerate(slots)}

    bi_label2id = {b_l: i for i, b_l in enumerate(bi_label)}
    id2bi_label = {i: b_l for i, b_l in enumerate(bi_label)}
    return ner_label2id, ner_id2label, bi_label2id, id2bi_label, slots2id, id2slots, slots


ner_label2id, ner_id2label, bi_label2id, id2bi_label, slots2id, id2slots, slots = label_process(slot_file)
