import os, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import copy


token_lens = [0, 1, 3, 5, 10, 20, 30]

@torch.no_grad()
def calculateInfillPerplexity(sentence, model, tokenizer, gpu):
    
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)

    logits = model(input_ids=input_ids, labels=input_ids).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    mu = (probs[0] * log_probs[0]).sum(-1)
    sigma = ((probs[0]) * torch.square(log_probs[0])).sum(-1) - torch.square(mu)

    all_ratios = {}
    for token_len in token_lens:
        all_ratios[token_len] = []

    token_count = input_ids.shape[-1]
    for i in range(token_count-1):

        # top p(w2 | w1)
        top_k_probs = torch.sort(probs[:, i, :], dim=-1, descending=True)

        token_id = input_ids[:, i+1]
        opt_token_id = top_k_probs.indices[:, 0]

        if token_id != opt_token_id:
            input_tokens_i = copy.deepcopy(input_ids)
            input_tokens_i[:, i+1] = opt_token_id

            logits_opt = model(input_ids = input_tokens_i, labels = input_tokens_i).logits
            probs_opt = torch.nn.functional.softmax(logits_opt, dim=-1)
            log_probs_opt = torch.nn.functional.log_softmax(logits_opt, dim=-1)

            mu_opt = (probs_opt[0] * log_probs_opt[0]).sum(-1)
            sigma_opt = ((probs_opt[0]) * torch.square(log_probs_opt[0])).sum(-1) - torch.square(mu)

            # w1 w2_training w3, w1 w2_most_likely w3
            # ratio = p(w2_training | w1 w3) / p(w2_most_likely | w1 w3)
            ratios = {}

            for token_len in token_lens:
                ratios[token_len] = ((log_probs[:, i, token_id] - mu[i]) / sigma[i].sqrt()) - ((log_probs_opt[:, i, opt_token_id] - mu_opt[i]) / sigma_opt[i].sqrt()) 

            for token_len in token_lens:
                for j in range(i, min(i+token_len, token_count-3)):
                    next_token_id = input_ids[:, j+2]
                    r =  ((log_probs[:, j+1, next_token_id] - mu[i]) / sigma[i].sqrt()) - ((log_probs_opt[:, j+1, next_token_id] - mu_opt[i]) / sigma_opt[i].sqrt())
                    ratios[token_len] = ratios[token_len] + r
                all_ratios[token_len].append(ratios[token_len].item())
        else:
            for token_len in token_lens:
                all_ratios[token_len].append(0.)
        
    return all_ratios

# helper function
def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='EleutherAI/pythia-2.8b')
parser.add_argument(
    '--dataset', type=str, default='WikiMIA_length32', 
    choices=[
        'WikiMIA_length32', 'WikiMIA_length64', 'WikiMIA_length128', 
        'WikiMIA_length32_paraphrased',
        'WikiMIA_length64_paraphrased',
        'WikiMIA_length128_paraphrased', 
    ]
)
parser.add_argument('--half', action='store_true')
parser.add_argument('--int8', action='store_true')
args = parser.parse_args()

# load model
def load_model(name):
    int8_kwargs = {}
    half_kwargs = {}
    if args.int8:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif args.half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    
    if 'mamba' in name:
        try:
            from transformers import MambaForCausalLM
        except ImportError:
            raise ImportError
        model = MambaForCausalLM.from_pretrained(
            name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
        )        
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

model, tokenizer = load_model(args.model)

# load dataset
if not 'paraphrased' in args.dataset:
    dataset = load_dataset('swj0419/WikiMIA', split=args.dataset)
else:
    dataset = load_dataset('zjysteven/WikiMIA_paraphrased_perturbed', split=args.dataset)
data = convert_huggingface_data_to_list_dic(dataset)

# inference - get scores for each input
scores = defaultdict(list)
for i, d in enumerate(tqdm(data, total=len(data), desc='Samples')): 
    text = d['input']
    
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    ll = -loss.item() # log-likelihood

    # assuming the score is larger for training data
    # and smaller for non-training data
    # this is why sometimes there is a negative sign in front of the score
    
    # loss and zlib
    scores['loss'].append(ll)
    scores['zlib'].append(ll / len(zlib.compress(bytes(text, 'utf-8'))))

    # mink and mink++
    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

    ## mink
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k_length = int(len(token_log_probs) * ratio)
        topk = np.sort(token_log_probs.cpu())[:k_length]
        scores[f'mink_{ratio}'].append(np.mean(topk).item())
    
    ## mink++
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu())[:k_length]
        scores[f'mink++_{ratio}'].append(np.mean(topk).item())
    
    ## ours: InfillMIA
    infill_probs = calculateInfillPerplexity(text, model, tokenizer, model.device)

    for token_ind in infill_probs:
        infill_prob = infill_probs[token_ind]
        infill_prob = np.nan_to_num(infill_prob)

        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(len(infill_prob) * ratio)
            topk = np.sort(infill_prob)[:k_length]
            scores[f'InfillMIA_{token_ind}tokens_{ratio}'].append(np.mean(topk).item())

# compute metrics
# tpr and fpr thresholds are hard-coded
def get_metrics(scores, labels):
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    return auroc, fpr95, tpr05

labels = [d['label'] for d in data] # 1: training, 0: non-training
results = defaultdict(list)
for method, scores in scores.items():
    auroc, fpr95, tpr05 = get_metrics(scores, labels)
    
    results['method'].append(method)
    results['auroc'].append(f"{auroc:.1%}")
    results['fpr95'].append(f"{fpr95:.1%}")
    results['tpr05'].append(f"{tpr05:.1%}")

df = pd.DataFrame(results)
print(df.to_string())

save_root = f"results/{args.dataset}"
if not os.path.exists(save_root):
    os.makedirs(save_root)

model_id = args.model.split('/')[-1]
if os.path.isfile(os.path.join(save_root, f"{model_id}.csv")):
    df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False, mode='a', header=False)
else:
    df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False)
