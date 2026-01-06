## 🚀 KG-Reasoner

**KG-Reasoner** is an end-to-end framework that enhances large language models with **structured multi-hop reasoning over Knowledge Graphs (KGs)**.

Unlike traditional pipeline-based KBQA systems, KG-Reasoner trains a Reasoning LLM to **directly perform KG traversal as part of its internal reasoning process**. By leveraging **reinforcement learning (RL)**, the model learns to:

- Dynamically explore multi-hop reasoning paths over KGs  
- Maintain global reasoning coherence across multiple steps  
- Perform backtracking when intermediate decisions lead to dead ends  

KG-Reasoner is evaluated on **eight challenging knowledge-intensive reasoning benchmarks** and consistently **matches or outperforms state-of-the-art approaches**, demonstrating strong effectiveness and robustness across diverse reasoning tasks.



### Package Directory Structure
```
.
├── requirements.txt
├── kg-tool
│   ├── config.py
│   ├── gnn_retrieval.py
│   ├── prompt_list.py
│   ├── kg_search_function.py
│   ├── utils.py
│   └── backtrack.py
├── README.md
├── OpenRLHF-RAG
├── scripts
│   ├── reinforce_train.sh
│   └── ray_start.sh
├── evaluation
│   ├── extract_entity_from_query.py
│   └── eval_search_kg.py
├── reward-remote
│   └── reward_server.py
└── data
    ├── training_set
    └── test_set
```
---

## Project Structure

- **`OpenRLHF-RAG/`**  
  Contains tools for RLHF (Reinforcement Learning with Human Feedback) model training.  
  This folder is adapted from [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).

- **`data/`**  
  Stores the training and testing datasets.

- **`evaluation/`**  
  Used to load trained models and perform evaluation.

- **`kg-tool/`**  
  Provides utilities for retrieving information from a knowledge graph.

- **`reward-remote/`**  
  Implements the remote reward function used during RL training.

- **`scripts/`**  
  Contains training scripts used in the RL training pipeline.

---

## Requirements
- Python 3.x
- Install the required libraries:
  ```bash
  pip install -r requirements.txt
## Usage

1. Enter the **KG-Hopper** folder:
   ```bash
   cd KG-Hopper
   ```
2. Training：
   ```bash
    ## Ray start
    bash scripts/ray_start.sh

    ## Start Reward Server
    python reward-remote/reward_server.py --port 1278

    ## Training
    bash scripts/reinforce_train.sh
    ```
3. Evaluation：
   ```bash
    python evaluation/eval_search_kg.py
   ```

