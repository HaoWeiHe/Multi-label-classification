from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from violation.ModelDefine import MultiLabel

class ViolationModel():
    def __init__(self):  
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.batch_size = 64

      model = MultiLabel()
      # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      model.load_state_dict(torch.load("model/bert_classifier.dict",map_location=torch.device('cpu')))
      #
      self.bert_classifier = model#torch.load("model/bert_classifier.pkl", map_location = "cpu").to(self.device)
      self.MAX_LENGTH =  30 
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def get_predictions(self, model, dataloader, compute_acc = False, sigmoid = True, thred = 0.5):
        predictions = None
        correct = 0
        total = 0
        acc_vals = []
        prods = []
        auc_valus = []
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)
                logits = model(b_input_ids,
                                b_attn_mask)
                if sigmoid: 
                  logits = logits.sigmoid()
                pred = (logits > 0.5).float().cpu().numpy()   
                prods = logits.float().cpu().numpy()   
                y_true = b_labels.byte().float().cpu().numpy()
                if compute_acc:
                  from sklearn.metrics import accuracy_score
                  acc = accuracy_score(y_true, pred)
                  from sklearn.metrics import roc_curve, auc
                  fpr = dict()
                  tpr = dict()
                  roc_auc = dict()
                  logits = logits.float().cpu().numpy()
                  for i in range(6):
                      fpr[i], tpr[i], _ = roc_curve(y_true[:, i], logits[:, i])
                      roc_auc[i] = auc(fpr[i], tpr[i])
                  # Compute micro-average ROC curve and ROC area
                  fpr["micro_avg"], tpr["micro_avg"], _ = roc_curve(y_true.ravel(), logits.ravel())
                  roc_auc["micro_avg"] = auc(fpr["micro_avg"], tpr["micro_avg"])
                  auc_valus.append(roc_auc["micro_avg"])
                  from sklearn.metrics import classification_report
                  acc_vals.append(acc)
            if predictions is None:
                predictions = pred, prods
            else:
                predictions = torch.cat((predictions, pred))
        if compute_acc:
            acc_vals = np.mean(acc_vals)
            val_auc_roc = np.mean(auc_valus)
            return predictions, acc_vals
        return predictions
      

    def preprocessing_for_bert(self, data):
        input_ids = []
        attention_masks = []
        
        for sent in data:
            encoded_sent = self.tokenizer.encode_plus(
                text = sent,  
                add_special_tokens = True,        # Add `[CLS]` and `[SEP]`
                max_length = self.MAX_LENGTH,                  
                pad_to_max_length = True,        
                #return_tensors='pt',           # Return PyTorch tensor
                truncation = True,
                return_attention_mask = True      # Return attention mask
                )
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        return input_ids, attention_masks


    def demo(self, stn = None, thred = 0.5, get_preds = True, get_probs = False):
        sample_input, sample_masks  = self.preprocessing_for_bert([stn])
        sample_labels = torch.tensor([1])
        sample_data = TensorDataset(sample_input, sample_masks, sample_labels)
        sample_dataloader = DataLoader(sample_data, batch_size=self.batch_size)
        preds, probs = self.get_predictions(self.bert_classifier, sample_dataloader, compute_acc = False, thred = thred)
        cls = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        preds = {x:y for x,y in zip(cls,preds[0].tolist())}
        probs = {x:y for x,y in zip(cls,probs[0].tolist())}
        if get_probs:
            return probs
        else:
            return preds
        




if __name__ == '__main__':

    stn = "how are you?"
    violation = ViolationModel()
    preds = violation.demo(stn,thred = 0.4)
    print(preds)
    

