# --- coding:utf-8 ---
import torch.nn as nn
from transformers import BertModel


class BiModel(nn.Module):
    def __init__(self,args):
        super(BiModel,self).__init__()

        self.model = BertModel.from_pretrained(args.pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        cls_out = outputs['pooler_output']

        output = self.dropout(cls_out)
        output = self.mlp(output)
        return output