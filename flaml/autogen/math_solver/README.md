# Math Solver

## Run

1. Set up env

```
pip install -e .[math] flaml
```

2. In `main.py`: set openai_key and wolfram id

```
openai.key = "Your key here"
os.environ["WOLFRAM_ALPHA_APPID"] = "Your id here"
```

3. Test out `main.py` with `--test_run`

```
cd flaml/autogen/math_solver
python main.py --prompt_type select --test_run
```

```
python main.py \
  --prompt_type ['select', 'python', 'wolfram']  \
  --max_round [default=15] \
  --use_cache [default=True] \
  --folder [default='./autotools'] \
  --cache_folder [default='./cache'] \
  [--test_run] # test run
```

4. Check results from path `saving_folder` (default is './autotools).

## Implementation

- `QueryHandler.py`:

  - Function `handle_query`:
    1. Parse all queries given an input string.
    2. Iterate over queries and call python or wolfram
    3. Return all results and a boolean indicating whether
- `MathSolver.py`: Main solver using tools.

  - Setting:

    - `use_cache=True`
    - `oai.ChatCompletion.request_timeout = 60*10`
    - `max_round=15`: max round of messages allowed
    - `len(query_response) < 2000`: The response should have less than 2000 chars (around 600-1000 tokens). To prevent excessive decimal numbers from python code.
    - `max_invalid_q_per_step=3`: For one step, if we keep getting invalide results for 3 times, we ask the LLM to solve the query itself.
  - Function `make_conversation`: get response from openai, extract query from response and get results

    - Answer is valid if '\box' is detected.
    - Answer is invalid (return empty string) if
      - response=-1
      - exceed max_round
      - exceed max token (8192 for GPT-4)
      - char count of query reply > 2000
  - Function `solve_one_category`: Solve problems from one category.

    - Assumption 1: when called with a problem set, all problems are of the same type
    - Assumption 2: if resume from a previous run, the sequence of problems from one category are the same as the previous run. This should be fine as long as the same shuffling seed is used.

## Prompts

'select' allows the model to choose from two tools, 'python' and 'wolfram' corresponding to one tool only.

select:

```
"""
Let's use two tools (python code and Wolfram alpha) to solve this problem step by step. You should always follow your own reasoning and only query when necessary.

First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through python or Wolfram alpha (for example, any calculations or equations that can be calculated) and choose the best tool to be used.
Please format the query in json: 
{ "tool" : "", # "python" or "wolfram"
"query": "", # your query here, either python code or Wolfram query.
} 
Note: when you put python code in the query, you should: 1.make sure the indentation is correct(use '\\t'). 2. use 'print' function for the output. 3. always use fractions instead of decimal.
4. Wait for me to give the results.
5. Correct this step based on the results, or give a new query if the results are invalid.
6. When you get the answer, put the answer in \\box{}.

Problem: 
"""
```

python:

```
"""
Let's use python code to solve this problem step by step. You should always follow your own reasoning and only query when necessary.

First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through python (for example, any calculations or equations that can be calculated). When you are querying python, you should: 1.use tab('\\t') for indentation. 2. use 'print' function for the output. 3. always output fractions instead of decimal.
Please format the query in json: 
{ "tool" : "python", 
"query": "", # your code here.
} 
4. Wait for me to give the results.
5. Correct this step based on the results, or give a new query if the results are invalid.
6. When you get the answer, put the answer in \\box{}.

Problem: 
"""
```

wolfram:

```
"""
Let's use Wolfram Alpha to solve this problem step by step. You should always follow your own reasoning and only query when necessary.

First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through Wolfram Alpha (for example, any calculations or equations that can be calculated). 
Please format the query in json: 
{ "tool" : "wolfram", 
"query": "", # your query here. Please use wolfram language.
} 
4. Wait for me to give the results.
5. Correct this step based on the results, or give a new query if the results are invalid.
6. When you get the answer, put the answer in \\box{}.

Problem:
"""
```
