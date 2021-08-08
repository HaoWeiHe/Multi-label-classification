from transformers import BertModel
import torch
import torch.nn as nn

class MultiLabel(nn.Module):
    def __init__(self, freeze_bert = False):
        super(MultiLabel, self).__init__()
        self.num_labels = 6
        D_in = 768 #config.hidden_size (of bert)
        H = 30  #hidden size of our model 
        
        self.DROPOUT = 0.5 #config.hidden_dropout_prob
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(self.DROPOUT) 
        # self.classifier = torch.nn.Linear(D_in, self.num_labels)
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(H, self.num_labels)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
               
    def forward(self, tokens_tensors, masks_tensors):
           
        outputs = self.bert(input_ids = tokens_tensors,
                            attention_mask = masks_tensors)
        
        outputs = self.dropout(outputs[0][:, 0, :])
        logits = self.classifier(outputs)
        
        return logits
