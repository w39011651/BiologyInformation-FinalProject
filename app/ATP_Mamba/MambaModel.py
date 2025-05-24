import torch
from torch import nn
from transformers import AutoModelForMaskedLM

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

        if self.reduction == 'mean':
            # 排除 ignore_index 對應的樣本
            if self.ignore_index != -100: # 檢查是否使用 ignore_index
                 valid_mask = (labels != self.ignore_index)
                 if valid_mask.sum() > 0:
                      return loss[valid_mask].mean()
                 else:
                      return torch.tensor(0.0, device=loss.device) # 如果沒有有效樣本，損失為 0
            else:
                 return loss.mean()
        elif self.reduction == 'sum':
             if self.ignore_index != -100:
                  valid_mask = (labels != self.ignore_index)
                  return loss[valid_mask].sum()
             else:
                  return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError("reduction 必須是 'mean', 'sum' 或 'none'")
        
class LCPLMforSequenceLabeling(nn.Module):
    def __init__(self, model_path, num_labels=2):
        super(LCPLMforSequenceLabeling, self).__init__()
        self.lc_plm = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        self.classifier = nn.Linear(self.lc_plm.config.d_model, num_labels)
        #this classifier to determine this amino acid is fad binding site or not
        #self.loss_fn = nn.CrossEntropyLoss().to("cuda")
        self.loss_fn = focal_loss(alpha=None, gamma=2.0).to("cuda")

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.lc_plm(input_ids = input_ids, attention_mask = attention_mask, labels = labels, output_hidden_states = True)
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