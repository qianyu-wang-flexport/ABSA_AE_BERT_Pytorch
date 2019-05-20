import torch
import torch.nn as nn


class BERT_TE(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_TE, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.class_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        encoded_layer, _= self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)
        output = self.dropout(encoded_layer)
        logits = self.dense(output)
        return logits
