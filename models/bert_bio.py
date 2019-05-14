# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_BIO(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_BIO, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.bio_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        encoded_layer, _= self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)
        output = self.dropout(encoded_layer)
        logits = self.dense(output)
        return logits
