import numpy as np
import torch
import random
import os
import logging

from transformers.modeling_bert import BertForSequenceClassificationFeatures,BertForSequenceClassificationFeatures2
from transformers import(
AdamW,
BertForSequenceClassification,
BertConfig, 
DNATokenizer,
get_linear_schedule_with_warmup
)

from transformers import glue_processors as processors

from finetune_model import train, predict


logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    "dna": (BertConfig, BertForSequenceClassification, DNATokenizer),
    "dnafeatures": (BertConfig, BertForSequenceClassificationFeatures, DNATokenizer),
    "dnafeatures2": (BertConfig, BertForSequenceClassificationFeatures2, DNATokenizer)
}


cfg = {
    "data_dir":"data/leave_one_out_testing",
    "model_type":"dna",
    "model_name_or_path":"pretrained_model/checkpoint-38950",
    "task_name":"dnaprom",
    "max_seq_length": 23 ,
    "per_gpu_eval_batch_size":300 ,
    "per_gpu_train_batch_size": 200,
    "pred_batch_size":200,
    "learning_rate": 2e-4 ,
    "num_train_epochs": 70,
    "logging_steps": 1 ,
    "warmup_percent": 0.1 ,
    "hidden_dropout_prob": 0.1 ,
    "attention_probs_dropout_prob": 0.1,
    "weight_decay": 0.01 ,
    "n_samples_dataset": 1000,
    "save_total_limits": 1 ,
    "gradient_accumulation_steps": 3,
    "adam_epsilon":1e-8,
    "beta1":0.9,
    "beta2":0.999,
    "output_dir": "outputs/leave_one_out_testing",
    "patience": 15
    }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

set_seed(42)

# Prepare GLUE task
cfg["task_name"] = cfg["task_name"].lower()
if cfg["task_name"] not in processors:
    raise ValueError("Task not found: %s" % (cfg["task_name"]))
processor = processors[cfg["task_name"]]()
label_list = processor.get_labels()
num_labels = len(label_list)

# Load pretrained model and tokenizer
cfg["model_type"] = cfg["model_type"].lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[cfg["model_type"]]


config = config_class.from_pretrained(
    cfg["model_name_or_path"],
    num_labels=num_labels,
    finetuning_task=cfg["task_name"],
    cache_dir=None,
)

config.hidden_dropout_prob = cfg["hidden_dropout_prob"]
config.attention_probs_dropout_prob = cfg["attention_probs_dropout_prob"]



sgRNA_list = os.listdir(cfg["data_dir"])
input_path = cfg["data_dir"]
output_path = cfg["output_dir"]
aucpr_results = []
for sgRNA in sgRNA_list:
    print(sgRNA)

    tokenizer = tokenizer_class.from_pretrained(
        "dna7",
        do_lower_case=False,
        cache_dir=None,
    )

    model = model_class.from_pretrained(
        cfg["model_name_or_path"],
        from_tf=bool(".ckpt" in cfg["model_name_or_path"]),
        config=config,
        cache_dir=None,
    )

    model.to(device)
    # change datapath to specific sgRNA
    cfg["data_dir"] = os.path.join(input_path,sgRNA)
    cfg["output_dir"] = os.path.join(output_path,sgRNA)

    train(cfg,model,tokenizer)
    result = predict(cfg,model,tokenizer,pred_dir="test")
    aucpr = result["auc-pr"] 
    print(aucpr)
    aucpr_results.append(aucpr)


print(aucpr_results)
print(np.mean(aucpr_results))