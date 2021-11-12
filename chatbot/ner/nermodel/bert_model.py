import torch.nn as nn

from .downstream import SelfAttention, LSTM, CRF
from transformers import BertModel


class NerModel(nn.Module):
    def __init__(self, args):
        super(NerModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrained_model_name)
        self.d_model = self.bert.config.hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.ds_name = args.downstream
        self.classifier = nn.Linear(self.d_model, args.num_classes)
        self.model_name = f"{args.pretrained_model_name}-{args.downstream}"
        self.args = args
        assert args.downstream in ['linear', 'lstm', "san", "crf" ,"lstm-crf"], \
            f"downstream model {args.downstream} not in linear, lstm, san or crf"

        if args.downstream == "linear":
            pass
        elif args.downstream == "lstm":
            self.lstm = LSTM(
                d_model=self.d_model,
                hidden_dim=self.d_model,
                num_layers=args.num_layers,
                args=args)
        elif args.downstream == "san":
            self.san = SelfAttention(d_model=self.d_model,
                                     num_heads=args.num_heads,
                                     dropout=args.dropout)
        elif args.downstream == "crf":
            self.crf = CRF(num_tags=args.num_classes,
                           include_start_end_transitions=True)

        elif args.downstream == "lstm-crf":
            self.lstm = LSTM(
                d_model=self.d_model,
                hidden_dim=self.d_model,
                num_layers=args.num_layers,
                args=args)
            self.crf = CRF(num_tags=args.num_classes,
                           include_start_end_transitions=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        attention_mask:  mask on padding position.
        """
        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(token_type_ids.shape)
        # print(labels.shape)
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,  # 0 if padding
                            token_type_ids=token_type_ids)  # segment id
        # print("?????")
        if self.args.augument:
            represents = outputs.last_hidden_state
        outputs = self.dropout(outputs.last_hidden_state)

        if self.ds_name == "san":
            outputs = self.san(outputs.transpose(0, 1),
                               key_padding_mask=(attention_mask == 1))
            outputs = outputs.transpose(0, 1)
        elif self.ds_name.startswith("lstm"):
            outputs = self.lstm(outputs)

        # linear projector after lstm, self-attention and before crf
        outputs = self.classifier(outputs)

        if self.ds_name.endswith('crf'):
            loss = -self.crf(inputs=outputs,
                             tags=labels,
                             mask=attention_mask)
            return loss, outputs
        else:
            if self.args.augument:
                return outputs, represents
            else:
                return outputs