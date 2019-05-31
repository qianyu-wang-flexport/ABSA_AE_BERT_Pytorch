import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Layer_Att(nn.Module):
    def __init__(self):
        super(Layer_Att, self).__init__()
        self.input_dim=12
        self.weight=Parameter(torch.randn(self.input_dim,1))

    def forward(self,inputs):
        # print(self.weight)
        softmax_weight=F.softmax(self.weight,dim=0)
        output_matrix=inputs.matmul(softmax_weight)
        output=output_matrix.squeeze(-1)
        return output



class BERT_MULTILAYER(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_MULTILAYER, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.class_dim)
        self.layer_att=Layer_Att()

    def forward(self, inputs):

        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        encoded_layers, _= self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=True)
        encoded_matrix=torch.stack(encoded_layers,-1)
        output=self.layer_att(encoded_matrix)
        output = self.dropout(output)
        logits = self.dense(output)
        return logits
