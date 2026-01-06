import os
import sys
import json
import copy
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../kg-tool')))

from kg_search_function import wiki_retrieval, wiki_retrieval_with_backtrack
from .extract_entity_from_query import judge_entity

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--start_sample", type=int, default=-1)
    parser.add_argument("--end_sample", type=int, default=100000)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--src_file", type=str, default="None")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--model_path", type=str, default="None")
    parser.add_argument("--gpu_memory_rate", type=float, default=0.95)
    parser.add_argument("--port", type=str, default="None")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--prompt_type", type=str, default="None")
    return parser.parse_args()

def clear_file(file_path):
    try:
        with open(file_path, 'w') as f:
            f.write('')
        print(f"File '{file_path}' has been cleared.")
    except Exception as e:
        print(f"Failed to clear file '{file_path}': {e}")

base_prompts = {
    "default": """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, the Assistant can perform searching for uncertain knowledge from a knowledge graph if necessary with the format of "<search> search query (only entity) here </search>". **A query must involve only a single triple**.
Then, the system will provide the Assistant with helpful information with the format of "<triples> ...search results... <triples>".\n\nUser:{question}\nAssistant: <think>""",
    "slot_filling": """The User give a sentence with a slot [SEP], and the Assistant fill the slot.
...\n\nUser:{question}\nAssistant: <think>""",
    "fact_checking": """The User gives a claim, and the Assistant verifies the truthfulness of the claim.
...\n\nUser:{question}\nAssistant: <think>"""
}

def prepare_dataset(data_path, question_key):
    with open(data_path, encoding='utf-8') as f:
        return json.load(f)[14:], question_key

def generate_prompt(data_path, question):
    if "creak" in data_path:
        return base_prompts["fact_checking"].format(question=question)
    elif "T-REX" in data_path or "Zero_Shot_RE" in data_path:
        return base_prompts["slot_filling"].format(question=question)
    return base_prompts["default"].format(question=question)

def main(data_path_qs_list):
    print("=Begin=" * 10)
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_rate,
        trust_remote_code=True
    )

    for data_path, question_key in data_path_qs_list:
        write_path = f"{os.path.basename(data_path).split('.')[0]}-{os.path.basename(args.model_path)}_base_temp{args.temp}_type{args.prompt_type}_result.jsonl"
        clear_file(write_path)
        print("write_path:", write_path)

        datas, question_key = prepare_dataset(data_path, question_key)
        for data in tqdm(datas):
            try:
                time.sleep(1)
                question = data.get(question_key) or data.get("input") or data.get("sentence")
                answer = str(data.get("answer") or data.get("answers") or data.get("label") or '')

                if not question:
                    continue

                topic_entity = data.get("topic_entity") or data.get("qid_topic_entity")
                if not topic_entity:
                    continue
                topic_entity_list = list(topic_entity.values())

                prompt_think = generate_prompt(data_path, question)
                stop_tokens = ["<|im_end|>", "<|endoftext|>", "</search>", "</answer>"]
                sampling_params = SamplingParams(temperature=args.temp, top_p=0.95, max_tokens=512, stop=stop_tokens)

                for _ in range(10):
                    output = llm.generate(prompt_think, sampling_params)[0]
                    generated_text = output.outputs[0].text
                    stop_reason = output.outputs[0].stop_reason

                    if "<answer>" in generated_text and stop_reason == "</answer>":
                        result = {
                            "question": question,
                            "answer": answer,
                            "pred_ans": generated_text.split("<answer>")[-1].split("</answer>")[0],
                            "stop_reason_final": "finished",
                            "gen_text_store": prompt_think + generated_text + "</answer>"
                        }
                        with open(write_path, "a") as f:
                            f.write(json.dumps(result) + "\n")
                        break

                    elif "<search>" in generated_text and stop_reason == "</search>":
                        query = generated_text.split("<search>")[-1].split("</search>")[0].strip()
                        entity = judge_entity(query) or topic_entity_list[0]
                        # doc_content = '\n'.join(wiki_retrieval(question, entity))
                        doc_content = '\n'.join(wiki_retrieval_with_backtrack(question, entity))
                        prompt_think += f"</search>\n\n<triples>\n{doc_content}<triples>\n\n"
                    else:
                        result = {
                            "question": question,
                            "answer": answer,
                            "pred_ans": "I don't know.",
                            "stop_reason_final": "shot_down"
                        }
                        with open(write_path, "a") as f:
                            f.write(json.dumps(result) + "\n")
                        break

            except Exception as e:
                print("Error:", e)
                continue

if __name__ == "__main__":
    data_path_qs_list = []  # your data paths and question keys
    main(data_path_qs_list)
