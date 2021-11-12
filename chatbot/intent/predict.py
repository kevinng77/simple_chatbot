import torch
from transformers import BertTokenizer
from model.bert_model import IntentModel
import sys

sys.path.append("..")
import config


class IntentPrediction():
    def __init__(self, args, intent_weight):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.intent_model = IntentModel(args=args)
        self.intent_model.to(self.device)
        intent_state_dict = torch.load(intent_weight)
        self.intent_model.load_state_dict(intent_state_dict)
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
        self.args = args
        self.intent_model.eval()

    def intent_predict(self, content):
        contents = []
        for slot in config.slots:
            contents.append([content, '有在问' + slot + '吗？'])

        inputs = self.tokenizer(contents,
                                max_length=128,
                                return_tensors='pt',
                                padding=True
                                ).to(self.device)

        intent_pred = self.intent_model(**inputs)

        y_pred = torch.argmax(intent_pred, 1).cpu().numpy()

        intents = []
        for i, label in enumerate(y_pred):
            if label == 1:
                intents.append(config.id2slots[i])

        return intents


if __name__ == '__main__':
    args = config.get_args()
    predictor = IntentPrediction(args, intent_weight=config.intent_weight)
    result = predictor.intent_predict("你好")
    print(result)
