import torch
from torch import nn
from transformers import AutoModelForMaskedLM, EsmConfig
import numpy as np

class focal_loss(nn.Module):
    def __init__(self, alpha = None, gamma=2.0, reduction = 'mean', ignore_index = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self ,logits, labels):
        ce_loss = nn.functional.cross_entropy(logits, labels, reduce="mean", ignore_index=-100)
        pt = torch.exp(-ce_loss)
        focal_term = (1-pt)**self.gamma
        loss = focal_term * ce_loss

        if not self.alpha is None:
            alpha_tensor = torch.ones_like(labels, dtype=torch.float32)
            alpha_tensor[labels==1] = self.alpha
            alpha_tensor = alpha_tensor.to(loss.device)

            loss = alpha_tensor * loss

        if self.ignore_index is not None:
          valid_mask = (labels != self.ignore_index)
          loss = loss[valid_mask]
          labels = labels[valid_mask]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError("reduction 必須是 'mean', 'sum' 或 'none'")
        
class CustomClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels, fc_hidden_size = 500, dropout_rate = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.fc_hidden_size = fc_hidden_size
        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.fc_hidden_size)
        self.fc1_bn = nn.BatchNorm1d(self.fc_hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_out = nn.Linear(self.fc_hidden_size, self.num_labels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        batch, seq_len, hidden = x.shape
        x = x.view(-1, hidden)
        x = self.bn(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        x = x.view(batch, seq_len, -1)
        return x
        

class EsmForSequenceLabeling(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(EsmForSequenceLabeling, self).__init__()
        self.esm = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

        self.classifier = CustomClassifier(self.esm.config.hidden_size, num_labels)
        self.loss_fn = focal_loss(alpha=0.8, gamma=2.0).to("cuda")

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.esm(input_ids = input_ids, attention_mask = attention_mask, labels = labels, output_hidden_states = True)
        sequence_output = outputs.hidden_states[-1]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # mask = (labels != -100)
            # logits = logits[mask]
            # labels = labels[mask]
            # 重塑 logits 和標籤以適應 CrossEntropyLoss
            # logits 從 [batch_size, seq_length, num_classes] 變為 [batch_size * seq_length, num_classes]
            active_logits = logits.view(-1, logits.size(-1))
          # 標籤從 [batch_size, seq_length] 變為 [batch_size * seq_length]
            active_labels = labels.view(-1)

            loss = self.loss_fn(active_logits, active_labels)

        return loss, logits