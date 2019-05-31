import torch
import torch.nn as nn
from models.bert_multilayer import Layer_Att


class BERT_SATE_MUTILAYER(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SATE_MUTILAYER, self).__init__()
        self.bert = bert
        self.layer_att=Layer_Att()
        self.dropout1 = nn.Dropout(opt.dropout)
        self.dropout2 = nn.Dropout(opt.dropout)
        self.dense1 = nn.Linear(opt.bert_dim, opt.senti_class_dim)
        self.dense2 = nn.Linear(opt.bert_dim,opt.bio_class_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        encoded_layers, label_layer= self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=True)
        encoded_matrix = torch.stack(encoded_layers, -1)
        output2 = self.layer_att(encoded_matrix)
        output1 = self.dropout1(label_layer)
        output2=self.dropout2(output2)
        senti_output = self.dense1(output1)
        bio_output=self.dense2(output2)
        return senti_output,bio_output
