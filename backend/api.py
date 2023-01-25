import numpy as np
import torch
import time
from xlwt import Workbook
import pathlib

from transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                          BertTokenizer, BertForMaskedLM)
from .class_register import register_api

class AbstractLanguageChecker:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def check_probabilities(self, in_text, topk=40):
        raise NotImplementedError

    def postprocess(self, token):
        raise NotImplementedError

    def top_k_logits(logits, k):
        """
        Filters logits to only the top k choices
        from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
        """
        if k == 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1]
        return torch.where(logits < min_values,
                       torch.ones_like(logits, dtype=logits.dtype) * -1e10,
                       logits)

@register_api(name='gpt-2-small')
class LM(AbstractLanguageChecker):
    def __init__(self, model_name_or_path="gpt2"):
        super(LM, self).__init__()
        self.enc = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.start_token = self.enc(self.enc.bos_token, return_tensors='pt').data['input_ids'][0]
        print("Loaded GPT-2 model!")
    def check_probabilities(self, in_text, topk=40):
        # Process input
        l=[0,0,0,0]
        token_ids = self.enc(in_text, return_tensors='pt').data['input_ids'][0]
        token_ids = torch.concat([self.start_token, token_ids])
        # Forward through the model
        output = self.model(token_ids.to(self.device))
        all_logits = output.logits[:-1].detach().squeeze()
        # construct target and pred
        all_probs = torch.softmax(all_logits, dim=1)

        y = token_ids[1:]
        # Sort the predictions for each timestep
        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
        # [(pos, prob), ...]
        real_topk_pos = list(
            [int(np.where(sorted_preds[i] == y[i].item())[0][0])
             for i in range(y.shape[0])])
        # real_topk_probs = all_probs[np.arange(
        #     0, y.shape[0], 1), y].data.cpu().numpy().tolist()
        # real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))
        for i in real_topk_pos:
            if i>=1000:
                l[3]+=1
            elif i<1000 and i>=100:
                l[2]+=1
            elif i<100 and i>=10:
                l[1]+=1
            elif i<10:
                l[0]+=1
        
        result = Workbook()
        wk = result.add_sheet('sheet1')
        wk.write(0,0,'Top10')
        wk.write(0,1,'Top100')
        wk.write(0,2,'Top1000')
        wk.write(0,3,'Above1000')

        for i in range(len(l)):
            wk.write(1,i,l[i])
        result.save('C:/Users/Pavani Rangineni/Desktop/result.xls')
        
        bpe_strings = self.enc.convert_ids_to_tokens(token_ids[:])

        bpe_strings = [self.postprocess(s) for s in bpe_strings]

        topk_prob_values, topk_prob_inds = torch.topk(all_probs, k=topk, dim=1)

        pred_topk = [list(zip(self.enc.convert_ids_to_tokens(topk_prob_inds[i]),
                              topk_prob_values[i].data.cpu().numpy().tolist()
                              )) for i in range(y.shape[0])]
        pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]


        # payload = {'bpe_strings': bpe_strings,
        #            'real_topk': real_topk,
        #            'pred_topk': pred_topk}

        payload ={'result':l}
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(l)

        return payload
    
    def postprocess(self, token):
        with_space = False
        with_break = False
        if token.startswith('Ġ'):
            with_space = True
            token = token[1:]
            # print(token)
        elif token.startswith('â'):
            token = ' '
        elif token.startswith('Ċ'):
            token = ' '
            with_break = True

        token = '-' if token.startswith('â') else token
        token = '“' if token.startswith('ľ') else token
        token = '”' if token.startswith('Ŀ') else token
        token = "'" if token.startswith('Ļ') else token

        if with_space:
            token = '\u0120' + token
        if with_break:
            token = '\u010A' + token

        return token

def main():
    raw_text = """"""
    