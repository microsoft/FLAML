# MathChat: A conversational framework for math problem solving with GPT-4

## Introduction:

Employing Large Language Models (LLMs) to address mathematical problems is an intriguing research endeavor, with LLMs demonstrating remarkable proficiency in various tasks spanning diverse domains. We propose *MathChat*, a framework that simulates a mock conversation between an LLM assistant (GPT-4 in our case) and a user proxy agent. Here a user proxy agent is an agent playing the user's role in conversations with the LLM assistant. In *MathChat*, the assistant and the user proxy agent work together to solve the math problem.Here a user proxy agent is an agent playing the user's role in conversations with the LLM assistant. In *MathChat*, the assistant and the user proxy agent work together to solve the math problem (See Figure below). The user proxy agent takes a math problem to be solved as input and would initiate a conversation with the LLM assistant using an intial prompt. With proper modifications, effective prompting methods from existing research, such as CoT and tool-using, can be integrated into the *MathChat* framework.

More details are provided in our paper [An Empirical Study on Challenging Math Problem Solving with GPT-4](https://arxiv.org/abs/2306.01337).

## Environment Setup

1. You can set up the environment using the following commands:

```
cd flaml/autogen/math
conda env create -f environment.yml
conda activate mathchat
```

2. Create a `key.txt` file in `flaml/autogen/math`, and put your openai key in it. The key should allow GPT-4 usage.

```
echo "your_openai_key" > key.txt
```

3. If you want to try out the wolfram prompt, you need to register a wolfram id and put it in `wolfram.txt`, which will be read in `main.py`.

```
echo "your_wolfram_key" > wolfram.txt
```

## Run MathChat

- Use `--categories` to select category to run, and `--samples_per_category` for number of samples. The problems are randomly selected from level-5 difficulty. Here are the category names and IDs:
| ID | Category Name            |
|----|--------------------------|
| 0  | Algebra                  |
| 1  | Counting & Probability   |
| 2  | Geometry                 |
| 3  | Intermediate Algebra     |
| 4  | Number Theory            |
| 5  | Prealgebra               |
| 6  | Precalculus              |


- Test on 1 level-5 problem from Alegbra (`--categories 0`):

```python
python main.py -ptype default --folder ./default --categories 0 --samples_per_category 1
```
You can find the output in folder `./default/`.

- Test on 1 level-5 problem from each category (except geometry):
```python
python main.py -ptype default --folder ./default --categories 0 1 3 4 5 6 --samples_per_category 1
```

Note: `default` is the default prompt for *MathChat*, other choices are `python` and `two_tools`.

- Test on all problems from each category (except geometry):

```python
python main.py -ptype default --folder ./default --categories 0 1 3 4 5 6 --samples_per_category 400
```

Note that no category has more that 400 problems, by setting `--samples_per_category 400` will take all problems.

## Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{wu2023empirical,
    title={An Empirical Study on Challenging Math Problem Solving with GPT-4},
    author={Yiran Wu and Feiran Jia and Shaokun Zhang and Hangyu Li and Erkang Zhu and Yue Wang and Yin Tat Lee and Richard Peng and Qingyun Wu and Chi Wang},
    year={2023},
    booktitle={ArXiv preprint arXiv:2306.01337},
}
```
