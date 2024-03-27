import numpy as np
import torch
import random
import os
import logging
import json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, WeightedRandomSampler
from tqdm import tqdm, trange
from transformers.modeling_bert import BertForSequenceClassificationFeatures,BertForSequenceClassificationFeatures2
from transformers import(
AdamW,
BertForSequenceClassification,
BertConfig, 
DNATokenizer,
get_linear_schedule_with_warmup
)

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    "dna": (BertConfig, BertForSequenceClassification, DNATokenizer),
    "dnafeatures": (BertConfig, BertForSequenceClassificationFeatures, DNATokenizer),
    "dnafeatures2": (BertConfig, BertForSequenceClassificationFeatures2, DNATokenizer)
}


cfg = {
    "data_dir":"data/data_newsplit3_42",
    "model_type":"dna",
    "model_name_or_path":"pretrained_model/checkpoint-38950",
    "task_name":"dnaprom",
    "max_seq_length": 23 ,
    "per_gpu_eval_batch_size":300 ,
    "per_gpu_train_batch_size": 200,
    "pred_batch_size":200,
    "learning_rate": 2e-4 ,
    "num_train_epochs": 15,
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
    "output_dir": "outputs/simple_test",
    "patience": 15
    }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_cache_examples(cfg, task, tokenizer, evaluate=False,do_predict=False,pred_dir =""):

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        cfg["data_dir"],
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, cfg["model_name_or_path"].split("/"))).pop(),
            str(cfg["max_seq_length"]),
            str(task),
        ),
    )
    if do_predict==True:
        cached_features_file = os.path.join(
            cfg["data_dir"],pred_dir,
            "cached_{}_{}_{}".format(
                "dev",
                str(cfg["max_seq_length"]),
                str(task),
            ),
        )


    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", cfg["data_dir"])
        label_list = processor.get_labels()
        examples = (
            processor.get_dev_examples(cfg["data_dir"]) if evaluate else processor.get_train_examples(cfg["data_dir"])
        )   

        print("finish loading examples")

        # params for convert_examples_to_features
        max_length = cfg["max_seq_length"]
        pad_on_left = False
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id =  0


        
        features = convert_examples_to_features(
        examples,
        tokenizer,
        task=task,
        label_list=label_list,
        max_length=max_length,
        output_mode=output_mode,
        pad_on_left=pad_on_left,  # pad on the left for xlnet
        pad_token=pad_token,
        pad_token_segment_id=pad_token_segment_id,)
                
     
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    #dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    if cfg["task_name"] == "dnacrispr":
        all_feature_a_ids = torch.tensor([f.feature_a_ids for f in features], dtype=torch.long)
        all_feature_b_ids = torch.tensor([f.feature_b_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_feature_a_ids,all_feature_b_ids)
    else:

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset

def train(cfg,model,tokenizer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = load_and_cache_examples(cfg, cfg["task_name"], tokenizer, evaluate=False)
    train_weights = []     
    for i in range(len(train_dataset)):
        if train_dataset[i][3].item() == 0:
            train_weights.append(1/(122061-525))         
        else:
            train_weights.append(1/525)

    train_sampler = WeightedRandomSampler(weights=train_weights,num_samples=cfg["n_samples_dataset"],replacement=True)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg["per_gpu_train_batch_size"])


    t_total = len(train_dataloader) // cfg["gradient_accumulation_steps"] * cfg["num_train_epochs"]
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    sep_lr = ["feature_embeddings,feature_a_embeddings,feature_b_embeddings,feature_c_embeddings,feature_d_embeddings"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and n != "bert.embeddings.feature_embeddings.weight"],
            "weight_decay": cfg["weight_decay"],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in sep_lr)], "lr": 5e-4}
        #{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in sep_lr)], "lr": 2e-3}
    ]

    warmup_steps = int(cfg["warmup_percent"]*t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg["learning_rate"], eps=cfg["adam_epsilon"], betas=(cfg["beta1"],cfg["beta2"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    #Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", cfg["num_train_epochs"])
    logger.info("  Instantaneous batch size per GPU = %d", cfg["per_gpu_train_batch_size"])
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d", 1)
    logger.info("  Gradient Accumulation steps = %d", cfg["gradient_accumulation_steps"])
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(cfg["num_train_epochs"]), desc="Epoch")
    set_seed(42)  

    best_aucpr = 0
    stop_count = 0
    best_model_state_dict = None
    best_tokenizer = None       

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        probs = None   # reset list every epoch
        preds = None
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if cfg["task_name"] == "dnacrispr":
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "feature_a_ids": batch[4],"feature_b_ids": batch[5]}

            outputs = model(**inputs)
         
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            logits = outputs[1]
            if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            
            if cfg["gradient_accumulation_steps"] > 1:
                loss = loss / cfg["gradient_accumulation_steps"]

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if cfg["logging_steps"] > 0 and global_step % cfg["logging_steps"] == 0:
                    logs = {}

                    results, eval_acc, eval_loss, aucpr = evaluate(cfg, model, tokenizer)
  

                    for key, value in results.items():
                        eval_key = "eval_{}".format(key)
                        logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / cfg["logging_steps"]
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss


                    print(json.dumps({**logs, **{"step": global_step}}))


                    if cfg["patience"] != 0:
                     
                        if results["auc-pr"] <= best_aucpr:
                            stop_count += 1
                            print(f"############## stop count ({stop_count}) #############")
                
                        else:
                            best_aucpr = results["auc-pr"]
                            stop_count = 0
                            print(f"############# new best ({best_aucpr}) auc-pr #############")

                            #Save model checkpoint

                            # output_dir = os.path.join(cfg["output_dir"], "best_model")
                            # if not os.path.exists(output_dir):
                            #     os.makedirs(output_dir)
                            # model.save_pretrained(output_dir)
                            # tokenizer.save_pretrained(output_dir)

                            best_model_state_dict = model.state_dict()
                            best_tokenizer = tokenizer

                        if stop_count == cfg["patience"]:
                            logger.info("Early stop")


                            output_dir = os.path.join(cfg["output_dir"], "best_model")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model.load_state_dict(best_model_state_dict)
                            model.save_pretrained(output_dir)
                            best_tokenizer.save_pretrained(output_dir)
                            return 





    
    if best_model_state_dict is not None and best_tokenizer is not None:
        output_dir = os.path.join(cfg["output_dir"], "best_model")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    return 

def evaluate(cfg, model, tokenizer, prefix="", evaluate=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_task = cfg["task_name"]
    softmax = torch.nn.Softmax(dim=1)    
    results = {}
    eval_dataset = load_and_cache_examples(cfg, eval_task, tokenizer, evaluate=evaluate)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=cfg["per_gpu_eval_batch_size"])

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", cfg["per_gpu_eval_batch_size"])
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    probs = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if cfg["task_name"] == "dnacrispr":
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "feature_a_ids": batch[4],"feature_b_ids": batch[5]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    
    
    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(eval_task, preds, out_label_ids, probs)
    acc = result["acc"]
    aucpr = result["auc-pr"]

    results.update(result)
    logger.info("***** Eval results {} *****".format(prefix))
    eval_result = ""
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        eval_result = eval_result + str(result[key])[:5] + " "


    return results,acc,eval_loss,aucpr



def predict(cfg,model, tokenizer, pred_dir, prefix=""):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    softmax = torch.nn.Softmax(dim=1)
    

    predictions = {}
    pred_task = cfg["task_name"]
    pred_dataset = load_and_cache_examples(cfg, pred_task, tokenizer, evaluate=True,do_predict=True,pred_dir=pred_dir)
    

 
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=cfg["pred_batch_size"])

    # Eval!
    logger.info("***** Running prediction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", cfg["pred_batch_size"])
    pred_loss = 0.0
    nb_pred_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(pred_dataloader, desc="Predicting"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if cfg["task_name"] == "dnacrispr":
                #inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "feature_a_ids": batch[4]}
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "feature_a_ids": batch[4],"feature_b_ids": batch[5]}
                #inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],"features_comb": batch[4]}
            outputs = model(**inputs)
            _, logits = outputs[:2]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)


    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(pred_task, preds, out_label_ids, probs)
    print(result)

    # pred_output_dir = args.predict_dir
    # if not os.path.exists(pred_output_dir):
    #         os.makedir(pred_output_dir)
    # output_pred_file = os.path.join(pred_output_dir, "pred_results.npy")

    logger.info("***** Pred results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    # np.save(output_pred_file, probs)
    
    return result



def main():
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

    train(cfg,model,tokenizer)
    predict(cfg,model,tokenizer,pred_dir="hek293t_test")
    predict(cfg,model,tokenizer,pred_dir="k562_test")

    return


if __name__ == "__main__":
    main()