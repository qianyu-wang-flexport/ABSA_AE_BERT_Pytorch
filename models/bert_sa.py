import torch
import torch.nn as nn


class BERT_SA(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SA, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.class_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, cls_layer= self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)
        output = self.dropout(cls_layer)
        logits = self.dense(output)
        return logits
