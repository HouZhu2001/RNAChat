import argparse
import os
import random
import time
import math
######## HF CACHE (LOAD BEFORE HF PACKAGES) ########
# os.environ['HF_HOME'] = "/data1/mingjia/cache/huggingface"
# print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
from rnachat.common.config import Config
from rnachat.common.registry import registry
from rnachat.common.dist_utils import get_rank, init_distributed_mode
from rnachat.common.conversation import Chat, CONV_VISION

from eval import get_simcse, get_simcse_llm_param
import json

# imports modules for registration
from rnachat.datasets.builders import *
from rnachat.models import *
from rnachat.runners import *
from rnachat.tasks import *

import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.",
                        default='configs/rnachat_eval.yaml')
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
init_distributed_mode(cfg.run_cfg)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

# # 先加载 stage-1 全量权重
# ckpt_stage1 = torch.load(cfg.model_cfg.stage1_ckpt, map_location='cpu')
# model.load_state_dict(ckpt_stage1['model'], strict=False)

# # 再加载 stage-2 LoRA 权重（或者用 peft 的 merge 方法）
# ckpt_stage2 = torch.load(cfg.model_cfg.peft_ckpt, map_location='cpu')
# model.load_state_dict(ckpt_stage2, strict=False)

chat = Chat(model, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list

def upload_rna(seq):
    chat_state = CONV_VISION.copy()
    img_list = []
    rna_emb, llm_message = chat.upload_rna(seq, chat_state, img_list)
    return chat_state, img_list, rna_emb

def gradio_ask(user_message, chat_state):
    chat.ask(user_message, chat_state)
    return chat_state

def gradio_answer(chat_state, img_list, num_beams=1, temperature=1e-3, top_p=0.9, save_embeds=False):
    # print(chat_state)
    llm_message, _, loss = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              top_p = top_p,
                              #repetition_penalty=2.0,
                              max_new_tokens=1500,
                              max_length=3200, 
                              save_embeds=save_embeds)
    return llm_message, chat_state, img_list, loss


if  __name__ == "__main__":

    print("Starting RNAChat Inference...")
    directory_name = "results"
    if not os.path.exists(directory_name):
        try:
            os.mkdir(directory_name)
        except Exception as e:
            print(f"An error occurred when creating results folder: {e}")

    df = pd.read_csv("rna_summary_2d.csv")
    ids = df['id'].values.tolist()[4200:]
    names = df['name'].values.tolist()[4200:]
    sequence = df['Sequence'].values.tolist()[4200:]
    labels = df['summary_no_citation'].values.tolist()[4200:]
    # ids = df['id'].values.tolist()[0:20]
    # names = df['name'].values.tolist()[0:20]
    # sequence = df['Sequence'].values.tolist()[0:20]
    # labels = df['summary_no_citation'].values.tolist()[0:20]
    func_text = []
    loss_list = []
    empty_predictions = []
    temperature = 0.3
    for i, (id, name, seq, lab) in enumerate(zip(ids,names, sequence, labels)):
        print(f"Processing {i+1}/{len(ids)}: {name}")
        if len(seq) > 1000:
            seq = seq[:1000]

        user_message = f"###Human: Give me a functional description of this RNA named {name}. ###Assistant:"
        chat_state, img_list, rna_embs = upload_rna(seq)
        chat_state = gradio_ask(user_message, chat_state)

        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=1, temperature=temperature)
    
        loss_list.append(loss)

        # # Test generation
        # user_message = "What is this RNA sequence? <RNAHere> Answer:"
        # test_samples = {
        #     'seq': [seq],
        #     'prompt': [user_message]
        # }
        # llm_message = model.generate(test_samples, max_length=1500, repetition_penalty=1.0)[0]

        # Check if prediction is empty or invalid
        if not llm_message or llm_message.strip() == "" or llm_message.strip() == "###Assistant:":
            print(f"⚠️  WARNING: Empty prediction for {name} (ID: {id})")
            empty_predictions.append({
                "id": id,
                "name": name,
                "seq": seq,
                "query": user_message,
                "correct_func": lab,
                "predict_func": llm_message,
                "loss": float(loss) if loss is not None else None
            })
            continue

        entry = {"seq": seq, "query": user_message, "correct_func": lab, "predict_func": llm_message}
        func_text.append(entry)
    
        print("Uniprot ID:", id)
        print("Correct summary:", lab)
        print(f"Predicted summary: {llm_message}")
        print('='*80)
    
    print("******************")
    print(f"Total samples processed: {len(ids)}")
    print(f"Valid predictions: {len(func_text)}")
    print(f"Empty predictions: {len(empty_predictions)}")
    
    # Save empty predictions for analysis
    if empty_predictions:
        empty_filename = f"results/empty_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(empty_filename, "w") as outfile:
            json.dump(empty_predictions, outfile, indent=4)
        print(f"Empty predictions saved to: {empty_filename}")
    
    # Only calculate metrics if we have valid predictions
    if not func_text:
        print("❌ ERROR: No valid predictions found! Cannot calculate metrics.")
        exit(1)
    
    print(f"Calculating metrics for {len(func_text)} valid predictions...")
    simcse_path = "princeton-nlp/sup-simcse-roberta-large"
    scores = get_simcse(simcse_path, func_text)
    
    output_filename = f"results/rnachat_inference_temperature{temperature}.json"
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj
    scores_serializable = convert_numpy_types(scores)
    with open(output_filename, "w") as outfile:
        json.dump(scores_serializable, outfile, indent=4)
    
    print(f"Results saved to: {output_filename}")
    print("RNAChat RNA Inference completed!")



