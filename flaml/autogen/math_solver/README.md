# Math Solver

## Run

1. Set up env

```
   pip install -e .[openai] flaml
   pip install datasets
```

2. In `main.py`: change openai_key
3. Run `main.py`. Default is set to test with 2 problems from each category

```
python flaml/autogen/math_solver/main.py
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
    - `max_tokens=4096`
    - `max_round=10`: allow 10 round of messages.
    - `len(query_response) < 2000`: The response should have less than 2000 chars (around 600-1000 tokens). To prevent excessive decimal numbers from python code.
    - Stop correcting queries from one step after 2 trials.
  - Function `make_conversation`: get response from openai, extract query from response and get results

    - Answer is valid if '\box' is detected.
    - Answer is invalid (return empty string) if
      - response=-1
      - exceed max_round
      - prompt_token + allowed_max_token > 8192.
      - char count of query reply > 2000
  - Function `solve_one_category`: Solve problems from one category.

    - Assumption 1: when called with a problem set, all problems are of the same type
    - Assumption 2: if resume from a previous run, the sequence of problems from one category are the same as the previous run. This should be fine as long as the same shuffling seed is used.

## Prompt

Currently using prompt that we manually tested and was proven to be effective. Some changes are made to better organize the prompt.

```
"""
Let's use two tools (python code and Wolfram Alpha) to solve this problem step by step. You should always follow your own reasoning and only query when necessary.

First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through python or Wolfram Alpha (for example, any calculations or equations that can be calculated) and choose the best tool to be used. When you are querying python, you should: 1.use tab('\\t') for indentation. 2. use 'print' function for the output. 3. always output fractions instead of decimal.
Please format the query in json: 
{ "tool" : "", # "python" or "wolfram"
"query": "", # your query here, either python code or Wolfram query.
} 
4. Wait for me to give the results.
5. Give a new query if the results are invalid or unexpected.
6. When you get the answer, put the answer in \\box{}.

Problem: 
"""
```
