import torch
from transformers import BertTokenizer
from utils.data_processer import label_process
from bimodel.bert_model import BiModel
import sys

sys.path.append("..")
import config

_, _, bi_label2id, id2bi_label, _, _, slots = label_process(config.slot_file)


class BiPrediction():
    def __init__(self, args, bi_weight):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.bi_model = BiModel(args=args)
        self.bi_model.to(self.device)
        bi_state_dict = torch.load(bi_weight)
        self.bi_model.load_state_dict(bi_state_dict)
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
        self.args = args
        self.bi_model.eval()

    def bi_predict(self, content):
        contents = []
        for slot in bi_label2id:
            contents.append([content, '对' + slot + '有要求吗'])

        inputs = self.tokenizer(contents,
                                return_tensors='pt',
                                max_length=128,
                                padding="max_length"
                                ).to(self.device)

        intent_pred = self.bi_model(**inputs)

        y_pred = torch.argmax(intent_pred, 1).cpu().numpy()

        bis = []
        for i, label in enumerate(y_pred):
            if label == 1:
                bis.append(id2bi_label[i])

        return bis


if __name__ == '__main__':
    args = config.get_args()
    predictor = BiPrediction(args, bi_weight=config.bi_weight)
    result = predictor.bi_predict("好的，北京万达文华酒店有吹风机吗？")
    print(result)
