import torch
import torch.nn as nn


class BERT_SA_TE(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SA_TE, self).__init__()
        self.bert = bert
        self.dropout1 = nn.Dropout(opt.dropout)
        self.dropout2 = nn.Dropout(opt.dropout)
        self.dense1 = nn.Linear(opt.bert_dim, opt.senti_class_dim)
        self.dense2 = nn.Linear(opt.bert_dim,opt.bio_class_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        encoded_layer, label_layer= self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)
        output1 = self.dropout1(label_layer)
        output2=self.dropout2(encoded_layer)
        senti_output = self.dense1(output1)
        bio_output=self.dense2(output2)
        return senti_output,bio_output
