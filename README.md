<p align="center">
   <img src="logo/logo.png" height="500%" width="40%" align='center' />
</p>

# 📖SEER: Facilitating Structured Reasoning and Explanation via Reinforcement Learning (ACL 2024)

This repository contains the code for our research paper titled "[SEER: Facilitating Structured Reasoning and Explanation via Reinforcement Learning](https://aclanthology.org/2024.acl-long.321/)", which has been accepted the main conference for ACL 2024.


<a href='https://arxiv.org/abs/2401.13246'><img src='https://img.shields.io/badge/Paper-Arxiv-red'> </a>


## News 📢

[10/01/2024] We have released the code.

[05/31/2024] **<a href="https://arxiv.org/abs/2401.13246.pdf"><b>Our Paper</b></a> is accepted at the main conference for ACL 2024**.


## TODO :pushpin:
- [ ] Release the checkpoint
- [x] Release the Code and Data
<!-- - [x] -->


# Usage :paperclip:

## Requirements
* Python 3.10
* Unbuntu 22.04
* Python Packages
``` bash
conda create -n seer python=3.10
conda activate seer
pip install -r requirements.txt
```

## Data
The data folder includes the EntailmentBank, EntailmentBankQA, STREET, eQASC and eOBQA.

See `data/Readme.md` for details.

Please download the retrieve, entailment and other modules to the `./exp`.

See `exp/Readme.md` for details.


## 1. supervised warm-up

### step_1: decompose structured reasoning
``` bash
python ./supervised_warm_up/preprocess_data/proof_to_step_data.py
python ./supervised_warm_up/preprocess_data/warm_state_data.py
```

### step_2: warm the policy up
``` bash
cd ./supervised_warm_up/scripts
bash task1.sh
```

**You can directly use `Policy for Task 3 (Iter0.zip) in ./exp` from FAME as the warmup model.**

## 2. RL phase

``` bash
cd ./etree_task1/scripts
bash SEER_task1.sh

cd ./etree_task2/scripts
bash SEER_task2.sh

cd ./etree_task3/scripts
bash SEER_task3.sh
```



# Introduction
## 1. An example of Structured Reasoning Task

<p align="center">
   <img src="images/example_of_entailment_tree.png" height="100%" width="40%" align='center' />
</p>



## 2. Method
### Overall framework of SEER

<p align="center">
   <img src="images/framework.png" height="100%" width="80%" align='center' />
</p>

> For trajectory rollout, action generation (Policy) and conclusion generation (entitlement) are performed alternately. The orange area details the reasoning process from st to st+1. For policy optimization, the reward module assigns rewards and updates the policy and critic based on tree or graph structures.


## 3. Experiments Results Analysis

### Experiment results on EntailmentBank (Table 1). 

**Bold and underlined texts highlight the best method and the runner-up**. RLET is based on DeBERTa-large, while all other methods are based on T5-large. All baseline results come from published papers. We use the `GPT-4-1106-preview` version for GPT-4.

<p align="center">
   <img src="table/1_table.png" height="100%" width="80%" align='center' />
</p>

<p align="center">
  <small>Table 1: Experiment results on EntailmentBank</small>


<!-- ### 🤗 Experiment results on the EntailmentBankQA (Table 2).

<p align="center">
   <img src="table/2_table.png" height="100%" width="40%" align='center' />
</p>

<p align="center">
  <small>Table 2: Experiment results on the EntailmentBankQA</small>

### 🤗 Experiment results on the STREET benchmark (Table 3).

<p align="center">
   <img src="table/3_table.png" height="500%" width="50%" align='center' />
</p>

<p align="center">
  <small>Table 3: Experiment results on the STREET benchmark.</small> -->


## 4. Illustrations

### Reward Process

<p align="center">
   <img src="images/reward_process.png" height="500%" width="100%" align='center' />
</p>


An illustration of the reward and alignment process of SEER. Each reasoning step represents a subtree.

(1) Tpred is constructed using the last intermediate conclusion (i4 in this example) as the hypothesis.

(2) The Jaccard similarity between the intermediate nodes (i∗) in Tpred and each golden intermediate node in Tgold (ˆi1 and h in this example) is calculated, and alignment is performed based on the maximum Jaccard similarity. In this example, i1 is aligned with ˆi1 because JS(i1,ˆi1) = 1. i2 is aligned with "NULL". i4 is aligned with ˆi1 because JS(i4,ˆi1) = 0.5 and JS(i4, h) = 0.4.

(3) Rewards are assigned based on the alignment results. Note that i3 (s3) is a redundant step. r1 = 1, r2 = -1, r3 = -0.5, and r4 = -1. The reward for each state originates from the tree structure rather than the chained trajectory. Therefore, the return of each state should also follow the tree structure (or graph structure in reasoning graphs) rather than the chained trajectory.


### Equivalent trajectory and Structure-based Return. 


<p align="center">
   <img src="images/equivalent_trajectory.png" height="500%" width="70%" align='center' />
</p>


An illustration of the equivalent trajectory and the definition of return. As the reasoning steps of $s_1$, $s_2$, and $s_3$ are mutually independent, the execution order among these steps can be arbitrary. Thus, $\tau$ and $\tau^{\prime}$ are equivalent trajectories because they can be converted to the same entailment tree. As shown in <font color=blue>blue box</font>, previous work defines the return (a.k.a cumulative reward) in a chained trajectory and would assign different returns to $s_1$ and $s_2$ in these equivalent trajectories. In contrast, as shown in <font color=red>red box</font>, our structure-based return is defined based on the tree or graph structure inherent in structured reasoning, which is the same source of rewards. Our structure-based return will consistently allocate stable returns to equivalent trajectories, thereby promoting training stability and convergence. Furthermore, maintaining consistency between the sources of rewards and returns can significantly enhance the effectiveness of the policy.


## 5. Citation

If our paper is helpful in your projects, please cite our paper and help to ⭐ this repo. Thanks.

```
@inproceedings{chen-etal-2024-seer,
    title = "{SEER}: Facilitating Structured Reasoning and Explanation via Reinforcement Learning",
    author = "Chen, Guoxin  and
      Tang, Kexin  and
      Yang, Chao  and
      Ye, Fuying  and
      Qiao, Yu  and
      Qian, Yiming",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.321",
    doi = "10.18653/v1/2024.acl-long.321",
    pages = "5901--5921",
}

```

## 6. Contact

If you have any questions, please raise an issue or contact us at 📧 Email: gx.chen.chn@gmail.com