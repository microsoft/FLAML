# Math Solver

## Run

1. Set up env

```
pip install -e .[math]
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

Arguments:

```
python main.py \
  --prompt_type ['select', 'python', 'wolfram']  \
  --max_round [default=15] \
  --folder [default='./autotools'] \
  --cache_folder [default='./cache'] \
  --samples_per_category [default=20] \
  --temperature [default=1, range[0,2]] \
  [--test_run] # test run
```

4. Check results from path `saving_folder` (default is './autotools).

### Baselines

1. Program of Thoughts (PoT)

```
cd flaml/autogen/math_solver
python baselines/PoT.py
```

Arguments:

```
python baselines/PoT.py \
  --folder [default='./PoT'] \
  --cache_folder [default='./cache/PoT'] \
  --samples_per_category [default=20] \
  [--dry_run] # output prompt with one problem from each category and do not query openai
```

## Implementation

- `QueryHandler.py`:

  - Function `handle_query`:
    1. Parse all queries given an input string.
    2. Iterate over queries and call python or wolfram
    3. Return all results and a boolean indicating whether all the queries are executed without error.
- `MathSolver.py`: Main solver using tools.

  - Setting:

    - `use_cache=True`
    - `oai.ChatCompletion.request_timeout = 60*10`
    - `max_round=15`: max round of messages allowed
    - `len(query_response) < 2000`: The response should have less than 2000 chars (around 600-1000 tokens). To prevent excessive decimal numbers from python code.
    - `max_invalid_q_per_step=3`: For one step, if we keep getting invalide results for 3 times, we ask the LLM to solve the query itself.
  - Function `make_conversation`: get response from openai, extract query from response and get results

    - Answer is valid if '\boxed{}' is detected.
    - Answer is invalid (return empty string) if
      - exceed max_round  (of conversations)
      - exceed max token (8192 for GPT-4)
      - char count of query reply > 2000
  - Function `solve_one_category`: Solve problems from one category.

    - Assumption 1: when called with a problem set, all problems are of the same type
    - Assumption 2: if resume from a previous run, the sequence of problems from one category are the same as the previous run. This should be fine as long as the same shuffling seed is used.

## Prompts

Three prompts available: ['select', 'python', 'wolfram'].
'select' allows the model to choose from two tools, 'python' and 'wolfram' corresponding to one tool only.

Please see `math_solver.py` for the prompts.
