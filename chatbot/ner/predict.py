import sys
import torch
from nermodel.bert_model import NerModel
from ner_utils.data_processer import Tokenizer


def load_model(model_path,  args, ner_label2id, ner_id2label):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(args=args,
                          ner_label2id=ner_label2id,
                          ner_id2label=ner_id2label)
    model = NerModel(args=args).to(args.device)

    model.load_state_dict(torch.load(model_path))
    return model, tokenizer, args


class NerPredicter():
    def __init__(self, model_path, ner_id2label,ner_label2id, args):
        self.model, self.tokenizer, self.args = load_model(model_path,
                                                           args=args,
                                                           ner_label2id=ner_label2id,
                                                           ner_id2label=ner_id2label)
        self.model.eval()
        self.ner_id2label = ner_id2label

    def ner_predict(self, content):
        inputs = self.tokenizer.bert_tokenizer(content,
                                               max_length=self.tokenizer.max_seq_len,
                                               return_tensors='pt',
                                               padding="max_length"
                                               )
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        if not self.args.downstream.endswith("crf"):
            outputs = self.model(**inputs)
            outputs = torch.masked_select(torch.argmax(outputs, dim=-1),
                                          inputs['attention_mask'] == 1).cpu().numpy()
        else:
            _, logits = self.model(**inputs, labels=inputs['attention_mask'])
            output = self.model.crf.viterbi_tags(logits=logits, mask=inputs['attention_mask'])
            outputs = torch.tensor(output, dtype=torch.long, device=self.args.device).view(-1).cpu().numpy()

        entities = []
        entity = ''
        for content, label in zip(content, outputs[1:-1]):
            label = int(label)
            if label == 0:
                if entity:
                    entities.append(entity)
                    entity = ''
                else:
                    continue
            else:
                if label % 2 == 1:
                    if entity:
                        entities.append(entity)
                    entity = self.ner_id2label[label].split('_')[1] + '\t' + content
                else:
                    if entity:
                        entity += content
                    else:
                        continue
        if entity:
            entities.append(entity)
        return entities


if __name__ == '__main__':
    sys.path.append("..")
    import config

    model = NerPredicter(model_path=config.ner_weight,
                         ner_id2label=config.ner_id2label,
                         args = config.get_args(),
                         ner_label2id=config.ner_label2id)
    content = "请在八达岭长城周边的景点给我推荐一个评分4.5分以上门票免费的景点游玩，谢谢了。"
    result = model.ner_predict(content)
    print(result)
