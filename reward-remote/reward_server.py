import argparse
import re
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import datasets
import random
from openrlhf.utils.logging_utils import init_logger
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import ujson as json
import re
import string
from collections import Counter
import pickle

logger = init_logger(__name__)


reasoning_reward_model_path = "REASONING_REWARD_MODEL_PATH"

reasoning_reward_tokenizer = AutoTokenizer.from_pretrained(reasoning_reward_model_path, trust_remote_code=True)
reasoning_reward_cal_model = AutoModelForCausalLM.from_pretrained(reasoning_reward_model_path, trust_remote_code=True, torch_dtype=torch.float16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reasoning_reward_cal_model = reasoning_reward_cal_model.to(device)

ans_eval_model_path = "ANS_EVAL_MODEL_PATH"  

ans_eval_tokenizer = AutoTokenizer.from_pretrained(ans_eval_model_path, trust_remote_code=True)
ans_eval_model = AutoModelForCausalLM.from_pretrained(ans_eval_model_path, trust_remote_code=True, torch_dtype=torch.float16)

ans_eval_model = ans_eval_model.to(device)



def extract_score(response: str) -> float:
    """Extract score from model response using regex."""
    match = re.search(r"score[:\s]*([01](?:\.\d+)?|0\.\d+)", response, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    return 0.0  # Default fallback


def evaluate_reasoning(question: str, reasoning_process: str) -> float:
    prompt = f"""
    You are an expert in reasoning tasks involving multi-turn retrieval interactions over knowledge graphs (KG). Your expertise includes decomposing complex questions into sub-queries and sequentially retrieving relevant knowledge from a KG to derive the correct answer.

    Your current task is to evaluate another agent's reasoning process for a given question, specifically determining whether their multi-turn retrieval and reasoning steps are logically sound. Given:

    A question about a knowledge topic.
    
    The reasoning process performed by the agent (including its decompositions, retrievals from the KG, and logical inferences).

    You must evaluate the reasonableness and logical coherence of this reasoning process. Provide a numerical score between 0 (completely unreasonable) and 1 (completely reasonable).
    The closer the score is to 1, the more reasonable the reasoning is.

    Evaluation and Score:

    Provide your evaluation clearly, followed by your final numerical score (0–1):

    Evaluation:

    Score:

    Begin your evaluation now:

    Question: {question}

    Reasoning Process: {reasoning_process}
    """
    inputs = reasoning_reward_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    generation_output = reasoning_reward_cal_model.generate(**inputs, max_new_tokens=256)
    response = reasoning_reward_tokenizer.decode(generation_output[0], skip_special_tokens=True)
    return extract_score(response)


def evaluate_answer(question: str, predicted: str, ground_truth: str) -> int:
    prompt = f"""
You are an expert in evaluating answer correctness for knowledge-based questions.

Your task is to determine whether the predicted answer given by a model is correct. You will be provided with:

a knowledge-based question,
the model’s predicted answer,
the ground truth answer.

Compare the predicted answer with the ground truth.
If the prediction is exactly correct, return '1'.
If the prediction is incorrect, return '0'.

Return only the value '1' or '0'—no explanation, no additional text.

Begin your evaluation now:

Question: {question}

Predicted Answer: {predicted}

Ground Truth: {ground_truth}

Your Answer:
"""

    inputs = ans_eval_tokenizer(prompt, return_tensors="pt").to(device)
    generation_output = ans_eval_model.generate(**inputs, max_new_tokens=5)
    response = ans_eval_tokenizer.decode(generation_output[0], skip_special_tokens=True)

    # "1" or "0"
    match = re.search(r"\b([01])\b", response)
    if match:
        return float(match.group(1))
    else:
        return float(normalize_text(predicted) == normalize_text(ground_truth))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s

def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(
        bool_mapping(ground_truth)
    )

def cover_exact_match_score_1(prediction, ground_truth):

    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")

    return all(ground in pre_list for ground in ground_list)

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def normalize_text(text):
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def extract_answer_math(s):
    return s.split("<answer>")[-1].split("</answer>")[0].strip()

def normalize_text(text):
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()


class MathRuleProxy:
    def __init__(self, args):
        eval_dataset = datasets.load_from_disk(args.data_path).to_list()
        self.eval_data_dict = self.get_answer_dict(eval_dataset)
        print(len(self.eval_data_dict))
        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain, trust_remote_code=True, use_fast=True)
        self.log_file = args.log_file
        self.cnt = 0

    def get_answer_dict(self, eval_dataset):
        eval_data_dict = {}
        for item in eval_dataset:
            eval_data_dict[normalize_text(item["question"])] = item["answer"]
        return eval_data_dict

    def get_qa(self, query):
        remove_prefix = " ".join(query.split("\n\nUser:")[1:])
        question = remove_prefix.split("\nAssistant: <think>")[0].strip()
        solution = query.split("\nAssistant: <think>")[-1].strip()
        return question, solution

    def get_query_answer(self, query):
        query = normalize_text(query)
        return self.eval_data_dict[query]

    def get_query_pred(self, query):
        return extract_answer_math(query)

    def get_reward(self, queries):
        preds = []
        answers = []
        questions = []
        solutions = []
        finished_lst = []
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
            print(queries[i])
            question, solution = self.get_qa(queries[i])
            preds.append(self.get_query_pred(solution))
            answers.append(self.get_query_answer(question))

            questions.append(question)
            solutions.append(solution)
        logger.info(f"queries[0]: {queries[0]}")
        # correct or not
        scores = []
        for t in range(len(queries)):
            f1_score_now, _ , _ = f1_score(preds[t], answers[t])
            scores.append(float(f1_score_now))
        
        for i in range(len(queries)):
            scores[i] += evaluate_answer(questions[i], preds[i], answers[i])
            scores[i] += evaluate_reasoning(questions[i], solutions[i])
        
        # format reward
        for i, query in enumerate(queries):
            self.cnt = self.cnt + 1
            if "<answer>" not in solutions[i] or "</answer>" not in solutions[i]:
                scores[i] = 0.0
                finished_lst.append("0")
            else:
                finished_lst.append("1")

            format_punishment=False
            count_1 = solutions[i].count("<triples>\n")
            count_2 = solutions[i].count("<triples>\n\n")
            count_3 = solutions[i].count("<search>")
            count_4 = solutions[i].count("</search>")
            count_5 = solutions[i].count("<triples>")
            count_6 = solutions[i].count("<triples>")
            count_7 = solutions[i].count("<triples>\n(1)")

            if count_1 == count_2 == count_3 == count_4 == count_5 == count_6 == count_7:
                pass
            else:
                format_punishment=True

            count_assiatant_1 = solutions[i].count("Assistant")
            count_assiatant_2 = solutions[i].count("assistant")
            if count_assiatant_1 == count_assiatant_2 ==0:
                pass
            else:
                format_punishment=True

            count_think_1 = solutions[i].count("<think>")
            count_think_2 = solutions[i].count("</think>")
            if count_think_1 ==0 and count_think_2==1:
                pass
            else:
                format_punishment=True

            count_answer_1 = solutions[i].count("<answer>")
            count_answer_2 = solutions[i].count("</answer>")
            if count_answer_1 == count_answer_2==1:
                pass
            else:
                format_punishment=True

            answer_text = solutions[i].split("<answer>")[-1].split("</answer>")[0].strip()
            if "begin_of_query" not in answer_text and "begin_of_documents" not in answer_text:
                pass
            else:
                format_punishment=True

            answer_len=len(answer_text.split())
            if answer_len > 10:
                format_punishment=True

            modified_solution = re.sub(r'<triples>>.*?<triples>>', '', solutions[i], flags=re.DOTALL)
            have_chinese = any('\u4e00' <= char <= '\u9fff' for char in modified_solution)
            if have_chinese:
                format_punishment=True


            if format_punishment==True:
                scores[i] = scores[i]-2

        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for q, a, s, f_f in zip(
                    questions,
                    solutions,
                    scores,
                    finished_lst,
                ):
                    record = {
                        "question": q,
                        "solution": a,
                        "score": s,
                        "finished": f_f,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--port", type=int, default=5001, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--log_file", type=str, default=None, help="Path to JSONL log file")

    args = parser.parse_args()

    # server
    reward_model = MathRuleProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
