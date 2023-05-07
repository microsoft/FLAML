# 1. Solve the problem step by step and do not overdivide the steps. Try to use python or wolfram to help you and aggregate as many steps as possible in one query.
# 2. I will take the queries and give the results.
# 3. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

# and try to use fractions to express your answer,
PROMPTS = {
    "v4.5python": """Let's use python to solve a math problem.

Query requirements:
You should always use 'print' function for the output, and use fractions/radical forms instead of decimal.
You must follow the formats below to write your code (otherwise it will not be recognized):
```python
# your code
```

First state the key idea to solve the problem. You may choose from 3 ways to solve the problem:
Case 1: If possible, write a program to directly solve it. If the problem involves enumerations, try to write a loop to iterate over all situations. Put your reasoning as comments in the code.
Case 2: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use python to check calculations if necessary.
Case 3: If the problem cannot be handled with the two ways above, please follow this process:
1. Solve the problem step by step (do not overdivide the steps).
2. Take out any queries that can be asked through python (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
    "v4.4python": """Let's solve a math problem with python.
First state what the problem is asking and the key idea to solve it. Then follow the process:
[Reasoning] # your reasoning
```python
# code to execute, print the output you want
```
[Reasoning] # your reasoning
```python
# code to execute, print the output you want
```
... (Until the problem is solved, or you need results to proceed)
Then I will help you execute the code and give you the results. If you need to keep solving the problem based on previous result, you can continue the process above.

Note: you should use exact representation of numbers instead of decimals and simplify the results (Use sympy).
Note: python is optional but recommended. You can directly solve the problem without python if the problem is mostly reasoning.
Note: correct the code according to the error message. If you keep getting error message, you should solve it yourself.
Note: an additional tool you can use is Wolfram Alpha. Put your query in
```wolfram
# wolfram query
```
and I will help you execute it.

After I give all results back to you, and you think the problem is finished, please reply "[EOF]".
""",
    "v4.3python": """Let's use python to solve a math problem.

First state the key idea to solve the problem. You may choose from 2 ways to solve the problem:
Case 1: If possible, write a program to directly solve it. If the problem involves enumerations, try to write a loop to iterate over all situations. Put your reasoning as comments in the code.
```python
# code. Put your reasoning as comments in the code.
```
Case 2: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use python to check calculations if necessary.
Case 3: If the problem cannot be handled with the two ways above, please follow this process:
[Reasoning1]
your reasoning
[Code]
```python
# put your code here. Remember to import used packages and define the variables. Please print the result you need
```
[Reasoning 2]
your reasoning
[Code]
```python
# code to execute, print the output you want
```
... (Keep going until the problem is solved, or you need results to proceed)
I will help you execute the code and give you the results. If you need to keep solving the problem based on previous result, you can continue the process above.
When you think the problem is finished, please reply "[EOF]".

Note: when using python, use exact representation of numbers instead of decimals and simplify the results.
""",
    "v4.2python": """Let's solve a math problem with python.
First state what the problem is asking and the key idea to solve it. Then follow the process:
[Reasoning] # your reasoning
```python
# code to execute, print the output you want
```
[Reasoning] # your reasoning
```python
# code to execute, print the output you want
```
... (Until the problem is solved, or you need results to proceed)
Then I will help you execute the code and give you the results. If you need to keep solving the problem based on previous result, you can continue the process above.
Note: you should use exact representation of numbers instead of decimals and simplify the results.
Note: python is optional but recommended. You can directly solve the problem without python if the problem is mostly reasoning.

After I give all results back to you, and you think the problem is finished, please reply "[EOF]".
""",
    "v4.1python": """Let's solve a math problem with python.
First state what the problem is asking and the key idea to solve it. Then follow the process:
[Reasoning] # your reasoning
```python
# code to execute, print the output you want
```
[Reasoning] # your reasoning
```python
# code to execute, print the output you want
```
... (Until the problem is solved, or you need results to proceed)
Then I will help you execute the code and give you the results. If you need to keep solving the problem based on previous result, you can continue the process above.
Note: you should use exact representation of numbers instead of decimals and simplify the results (Use sympy).

After I give all results back to you, and you think the problem is finished, please reply "[EOF]".
""",
    # Note: you should use exact representation of numbers instead of decimals and simplify the results (Use sympy).
    # v4
    "v4": """Let's use two tools (python code and Wolfram alpha) to solve a math problem. You can be flexible in choosing the approach or tools to solve the problem, but you are encouraged to use python or wolfram when necessary. It is best if we finish the problem in few rounds of conversations, but it also depends on the problem.

Query requirements:
You should choose the most suitable tool for each task. You must following the formats below to write your queries (otherwise it will not be recognized):
For python:
```python
# your code
```
For wolfram:
```wolfram
# your wolfram query
```
Note: 1. When writing python, use the 'print' function for the output, and use fractions/radical forms instead of decimal.
2. Wolfram might be suitable for symbolic manipulations (for example, simplifying expressions). Please use correct Wolfram language to describe your query. If you have several wolfram queries, put them in different code blocks.
3. Error handling: If the returned query result is invalid or unexpected, you can choose to 1. correct the query. 2. try another tool. 3. solve it yourself. Espically when you find the result is not correct or unexpected, you need to check your reasoning.

I will give you a math problem to solve. Below are some pieces of demonstrations that you can refer to:

# Ex 1
Agent:
```python
# your code... # you might choose to use python to solve the problem directly. You should include your reasoning in the comments.
```
User: # result
Agent: ...the answer is \\boxed{...}. # put final anwer in boxed.

#Ex2
Agent:
... # solving process, although encouraged to use tools, sometimes it is not necessary. So you can solve just solve the problem directly.
the answer is \\boxed{...}.

#Ex3
Agent: The key idea is ...
... # solving process
```python
# code, you decide to use python
```
...
User: # result
Agent:
... # you continue the solving process

#Ex4:you might have several queries
Agent: ...
```wolfram
# wolfram query # you wolfram code is separate from your python code, do not mix them together.
```
...
```python
# code, you python code can be continues
```
User:# result

# Ex 5:
Agent: ...
#query
...
User: #result
Agent: ...
#query
...
User: #result
...
""",
    "v3.4python": """Let's use python to solve a math problem.

Query requirements:
You should always use 'print' function for the output, and use fractions/radical forms instead of decimal.
You must follow the formats below to write your code:
```python
# your code
```

First state the key idea to solve the problem. You may choose from 3 ways to solve the problem:
Case 1: If possible, write a program to directly solve it. If the problem involves enumerations, try to write a loop to iterate over all situations. Put your reasoning as comments in the code.
Case 2: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use python to check calculations if necessary.
Case 3: If the problem cannot be handled with the two ways above, please solve the problem step by step. **Keep solving the problem. Only stop when you have queries and need to be executed.**
1. Output one step or several steps. (Do not overdivide the steps)
2. Take out any queries that can be asked through python (for example, any calculations or equations that can be calculated). Follow your own reasoning and query when necessary.
3. Wait for me to run the queries and return the results.
4. Correct this step based on the results, or give a new query if the results are invalid.

After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
    "v3.3python": """Let's use python to solve a math problem. You must use 'print' function for the output, and use exact numbers like radical forms instead of decimal (maybe use sympy). Follow this format to write your code:
```python
# your code
```

First state the key idea to solve the problem. You may choose from 3 ways to solve the problem:
Case 1: If possible, write one program to directly solve it (include your reasoning as comments). If the problem involves enumerations, try to write a loop to iterate over all situations.
Case 2: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use python to check calculations if necessary.
Case 3: If the problem cannot be handled with the two ways above, please follow this process:
1. Solve the problem step by step (do not overdivide the steps).
2. Take out any queries that can be asked through python (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
    # You should solve the problem and get to the answer directly if the problem only involve simple calculations or is mostly reasoning,
    # You should always use 'print' function for the output, and use exact numbers like radical forms instead of decimal (maybe use sympy).
    "v3.2python": """Let's use python to solve a math problem.

Query requirements for python:
You should always use 'print' function for the output, and use fractions/radical forms instead of decimal.
You must follow the formats below to write your code:
```python
# your code
```
When writing code, include your reasoning in the comments.

First state the key idea to solve the problem then choose from 3 ways to solve the problem:
Case 1: If possible, write a program to directly solve it. If the problem involves enumerations, try to write a loop to iterate over all situations. Put your reasoning as comments in the code.
Case 2: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use python to check calculations if necessary.
Case 3: If the problem cannot be handled with the two ways above, please follow this process:
1. Solve the problem step by step (do not overdivide the steps).
2. Take out any queries that can be asked through python (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
    # v3.7select from v3.6, set python to default
    "v3.7select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.

Query requirements:
You are provided with python code and Wolfram alpha to help you. By default you should use python but you can use Wolfram when it is more suitable.
Note: For code, you should always use 'print' function for the output, and use exact numbers (like radical forms) instead of decimal.
Note: For Wolfram, you should only have one query per code block, each query should be independent.
Following the format below (otherwise it will not be recognized):
For python:
```python
# your code
```
For wolfram:
```wolfram
# your wolfram query
```

First state the key idea to solve the problem. You may choose from 3 ways to solve the problem:
Case 1: If possible, write a program to directly solve it. If the problem involves enumerations, try to write a loop to iterate over all situations. Put your reasoning as comments in the code.
Case 2: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. It is good practice to use tools to help but not necessary.
Case 3: If the problem cannot be handled with the two ways above, please follow this process:
1. Solve the problem step by step (do not overdivide the steps).
2. Take out any queries that can be solved using python or Wolfram (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
    # v3.6select  v3.1python+wolfram, especially remove "Wolfram might be suitable for symbolic manipulations"
    "v3.6select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.

Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
Note: For code, you should always use 'print' function for the output, and use fractions/radical forms instead of decimal.
Note: For Wolfram, you should separate different queries in different code blocks.
Following the format below (otherwise it will not be recognized):
For python:
```python
# your code
```
For wolfram:
```wolfram
# your wolfram query
```

First state the key idea to solve the problem. You may choose from 3 ways to solve the problem:
Case 1: If possible, write a program to directly solve it. If the problem involves enumerations, try to write a loop to iterate over all situations. Put your reasoning as comments in the code.
Case 2: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. It is good practice to use tools to help but not necessary.
Case 3: If the problem cannot be handled with the two ways above, please follow this process:
1. Solve the problem step by step (do not overdivide the steps).
2. Take out any queries and choose the most suitable tool (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
    # v3.5select  v3python+wolfram
    "v3.5select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem. Your are provided with three ways to solve the problem, choose the best way to solve the problem and be flexible to switch to other ways if necessary.

Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must following the formats below to write your queries (otherwise it will not be recognized):
For python:
```python
# your code
```
For wolfram:
```wolfram
# your wolfram query
```
Note: When writing python, use the 'print' function for the output, and use fractions/radical forms instead of decimal.
Note: Wolfram might be suitable for symbolic manipulations (for example, simplifying expressions). Please use correct Wolfram language to describe your query. If you have several wolfram queries, put them in different code blocks.

First state the key idea to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use python to check calculations if necessary.
Case 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements above, and I will help you run it.
Case 3: If the problem cannot be handled with the above two ways, please follow this process:
1. Solve the problem step by step (do not overdivide the steps).
2. Take out any queries that can be asked through python or Wolfram (for example, any calculations or equations that can be calculated) and select the most suitable tool
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
    # Case 1: If possible, write a program to directly solve it. You should try this espically when it involves enumerations and you can write a loop to iterate over all situations.   Put your reasoning as comments in the code.
    "v3.1python": """Let's use python to solve a math problem.

Query requirements:
You should always use 'print' function for the output, and use fractions/radical forms instead of decimal.
You must follow the formats below to write your code (otherwise it will not be recognized):
```python
# your code
```

First state the key idea to solve the problem. You may choose from 3 ways to solve the problem:
Case 1: If possible, write a program to directly solve it. If the problem involves enumerations, try to write a loop to iterate over all situations. Put your reasoning as comments in the code.
Case 2: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use python to check calculations if necessary.
Case 3: If the problem cannot be handled with the two ways above, please follow this process:
1. Solve the problem step by step (do not overdivide the steps).
2. Take out any queries that can be asked through python (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
    # v3python only
    "v3python": """Let's use python to solve a math problem. Your are provided with three ways to solve the problem, choose the best way to solve the problem and be flexible to switch to other ways if necessary.

Query requirements:
When you write python code, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "python",
"query": "Your code here."
}

First state the key idea to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use python to check calculations if necessary.
Case 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements above, and I will help you run it.
Case 3: If the problem cannot be handled with the above two ways, please follow this process:
1. Solve the problem step by step (do not overdivide the steps).
2. Take out any queries that can be asked through python (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
    # change query format, change case 2
    # v1.6, come from v1.4, add wolfram note, add "follow your own reasoning"
    "v1.6select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem step by step.

Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must following the formats below to write your queries (otherwise it will not be recognized):
For python:
```python
# your code
```
For wolfram:
```wolfram
# your wolfram query
```
Note: When writing python, use the 'print' function for the output, and use fractions/radical forms instead of decimal.
Note: If you have several wolfram queries, please put different wolfram queries in different code blocks.

First state the key idea to solve the problem. Then follow the process:
1. Output one step (Do not overdivide the steps). Follow your own reasoning and query when necessary.
2. Take out any queries that can be asked through python or Wolfram alpha (for example, any calculations or equations that can be calculated) and choose the best tool to be used.
3. Wait for me to give the results.
4. Correct this step based on the results, or give a new query if the results are invalid.
When you get the answer, put the answer in \\boxed{}.
""",
    # too many rounds
    # Note: Wolfram might be more suitable for symbolic manipulations (such as simplifying expressions). If you query wolfram, you need to put different queries in different code blocks.
    "v1.5select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem step by step.

Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must following the formats below to write your queries (otherwise it will not be recognized):
For python:
```python
# your code
```
For wolfram:
```wolfram
# your wolfram query
```
Note: When writing python, use the 'print' function for the output, and use fractions/radical forms instead of decimal.
Note: If you have several wolfram queries, please put different wolfram queries in different code blocks.

First state the key idea to solve the problem. Then follow the process:
1. Output one step (Do not overdivide the steps). Follow your own reasoning and query when necessary.
2.
Case 1: If you have queries to wolfram or python, please parse the queries and wait for me to give the results.
Case 2: If there are no queries, go back to step 1 and continue.
3. Correct this step based on the results, or give a new query if the results are invalid.
When you get the answer, put the answer in \\boxed{}.
""",
    # v1.4select: new query format, add (Do not overdivide the steps)
    "v1.4select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem step by step.

Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must following the formats below to write your queries (otherwise it will not be recognized):
For python:
```python
# your code
```
For wolfram:
```wolfram
# your wolfram query
```
Note: When writing python, use the 'print' function for the output, and use fractions/radical forms instead of decimal.

First state the key idea to solve the problem. Then follow the process:
1. Output one step. (Do not overdivide the steps)
2. Take out any queries that can be asked through python or Wolfram alpha (for example, any calculations or equations that can be calculated) and choose the best tool to be used.
3. Wait for me to give the results.
4. Correct this step based on the results, or give a new query if the results are invalid.
When you get the answer, put the answer in \\boxed{}.
""",
    # v1.3select: only change the query format
    "v1.3select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem step by step.

Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must following the formats below to write your queries (otherwise it will not be recognized):
For python:
```python
# your code
```
For wolfram:
```wolfram
# your wolfram query
```
Note: When writing python, use the 'print' function for the output, and use fractions/radical forms instead of decimal.

First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through python or Wolfram alpha (for example, any calculations or equations that can be calculated) and choose the best tool to be used.
3. Wait for me to give the results.
4. Correct this step based on the results, or give a new query if the results are invalid.
When you get the answer, put the answer in \\boxed{}.
""",
    # best from v3.1 and 3.3
    "v3.4select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem. Your are provided with three ways to solve the problem, choose the best way to solve the problem and be flexible to switch to other ways if necessary.

Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the best tool from "python" or "wolfram".
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.

First state the key idea to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use tools to check your calculations when necessary.
Case 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements above. and I will help you execute it.
Case 3: If the problem cannot be handled with the above two ways, please solve the problem step by step following this process:
1. Output one step. (do not over divide the steps)
2. Take out any queries that can be asked with the tools (for example, any calculations or equations that can be calculated) and format your queries following the query requirements above.
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning, or choose a different tool.

After all the queries are executed and you get the answer, put the answer in \\boxed{}. Do not round up the answer unless it is required in the problem.
""",
    # v3.3 select
    "v3.3select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem. Your are provided with three ways to solve the problem, choose the best way to solve the problem and be flexible to switch to other ways if necessary.

Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
Note: Wolfram is suitable for symbolic manipulations (such as simplifying expressions).

First state the key idea to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem only involve simple calculations or is mostly reasoning, you can solve it by yourself directly. You can use tools to check your calculations when necessary.
Case 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements above, and I will help you run it.
Case 3: If the problem cannot be handled with the above two ways, please follow this process:
1. Solve the problem step by step (do not overdivide the steps).
2. Take out any queries that can be asked through python or Wolfram alpha (for example, any calculations or equations that can be calculated).
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.
""",
    # v3.1select
    "v3.1select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem. Your are provided with three ways to solve the problem, choose the best way to solve the problem and be flexible to switch to other ways if necessary.

Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.

First state the key idea to solve the problem and which way you would choose to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem is mostly reasoning and doesn't involve many calculations or symbol manipulations, you can solve it by yourself directly. You can use tools to check your answer if necessary.
Case 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements above. and I will help you execute it.
Case 3: If the problem cannot be handled with the above two ways, please follow this process:
1. Output one step. (do not over divide the steps)
2. Take out any queries that can be asked with the tools (for example, any calculations or equations that can be calculated) and format your query following the query requirements above.
3. Wait for me to give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning or choose a different tool.

After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
    # v3select
    "v3select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem. Your are provided with three ways to solve the problem, choose the best way to solve the problem and be flexible to switch to other ways if necessary.

Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
Note: Wolfram is suitable for symbolic manipulations (such as simplifying expressions).

First state the key idea to solve the problem. You may choose from three ways to solve the problem:
Case 1: If the problem is mostly reasoning and doesn't involve many calculations or symbol manipulations, you can solve it by yourself directly. If you suspect the result might be wrong, or you can use tools to check it.
Case 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements above. and I will help you execute it.
Case 3: If the problem cannot be handled with the above two ways, please follow this process:
1. Solve the problem step by step and do not overdivide the steps. Try to use python or wolfram to help you and aggregate as many steps as possible in one query.
2. I will take the queries and give the results.
3. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
    # "v2.1select" :
    "v2.1select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step and do not overdivide the steps. Try to use python or wolfram to help you and aggregate as many steps as possible in one query. In particular, if you think you can use one query to aggregate all steps to solve the problem, please do so.
Please follow the query requirements below, otherwise it will not be recognized:
    - Select the most suitable tool for the query.
    - Query python: put python code in ```python ... ```. You must 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
    - Query wolfram: put query ``wolfram ... ```. Note: Wolfram might be more suitable for symbolic manipulation and mathematical operations (such as simplifying expressions).
3. There should be one or more queries waiting to be executed. I will take the queries and give the results.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
    # v1.2select
    "v1.2select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.

First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step and do not overdivide the steps. Try to use python or wolfram to help you and aggregate as many steps as possible in one query. In particular, if you think you can use one query to aggregate all steps to solve the problem, please do so.
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
Note: Wolfram is suitable for symbolic manipulation and mathematical operations (such as simplifying expressions).
2. There should be one or more queries waiting to be executed. I will take the queries and give the results.
3. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
    # v1.1select
    "v1.1select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.

First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step. Do not overdivide the steps, and try to use python or wolfram to help you with one or more steps. If you think the problem can be solved with one query, please do so.
You must put the python code or wolfram query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the most suitable tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.
Note: Wolfram is suitable for symbolic manipulation and mathematical operations (such as simplifying expressions).
2. There should be one or more queries waiting to be executed. I will take the queries and give the results.
3. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
    # v2select  Try to use python or wolfram to help you with as many steps as possible. Choose the best tool for each task.
    "v2select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step. Try to use python or wolfram to help you with one or more steps. Choose the best tool to be used.
Follow this format:
    - When query python. put code in ```python ... ```. Always use fractions instead of decimal and use the 'print' function for the output.
    - When query wolfram, put query ``wolfram ... ```
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get the answer, put the answer in \\boxed{}.
""",
    # nostep
    "nostep": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.

First state the key idea to solve the problem. Then follow the process:
1. Try to use the tools to help you solve the problem. In particular, you can write a python program or wolfram query to solve the problem in one step if possible. Please use json format:
{ "tool" : "", #  select the best tool from "python" or "wolfram".
"query": "" # your query here, either python code or Wolfram query.
}
Note: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\\t'). 3. use the 'print' function for the output.
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get the answer, put the answer in \\boxed{}.
""",
    # v1select *** select *** good for user
    "v1select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step. Try to use python or wolfram to help you with one or more steps. Put the query in json:
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Note: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\\t'). 3. use the 'print' function for the output.
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get the answer, put the answer in \\boxed{}.
""",
    # *** select *** good for both system and user
    "select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem step by step. You should always follow your own reasoning and only query when necessary.

First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through python or Wolfram alpha (for example, any calculations or equations that can be calculated) and choose the best tool to be used.
Please format the query in json:
{ "tool" : "", # "python" or "wolfram"
"query": "" # your query here, either python code or Wolfram query.
}
Note: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\\t'). 3. use 'print' function for the output.
4. Wait for me to give the results.
5. Correct this step based on the results, or give a new query if the results are invalid.
6. When you get the answer, put the answer in \\boxed{}.
""",
    # use python
    "python": """Let's use python code to solve a math problem step by step. You should always follow your own reasoning and only query when necessary.

First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through python (for example, any calculations or equations that can be calculated). When you are querying python, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\\t'). 3. use 'print' function for the output.
Please format the query in json:
{ "tool" : "python",
"query": "" # your code here.
}
4. Wait for me to give the results.
5. Correct this step based on the results, or give a new query if the results are invalid.
6. When you get the answer, put the answer in \\boxed{}.
""",
    # use wolfram
    "wolfram": """Let's use Wolfram Alpha to solve a math problem step by step. You should always follow your own reasoning and only query when necessary.

First state the key idea to solve the problem. Then follow the process:
1. Output one step.
2. Take out any queries that can be asked through Wolfram Alpha (for example, any calculations or equations that can be calculated).
Please format the query in json:
{ "tool" : "wolfram",
"query": "" # your query here. Please use wolfram language.
}
4. Wait for me to give the results.
5. Correct this step based on the results, or give a new query if the results are invalid.
6. When you get the answer, put the answer in \\boxed{}.
""",
    # v1both
    "v1both": """Let's use two tools (python code and Wolfram alpha) to solve a math problem step by step. You should always follow your own reasoning and only query when necessary.

First state the key idea to solve the problem. Then follow the process:
1. Output one step. (do not overdivide the steps)
2. Take out any queries that can be asked through python or Wolfram alpha (for example, any calculations or equations that can be calculated).
You can query both tools for each task to cross-check the results. If you don't have query for one tool, just leave it blank.
Please format the query in json:
{ "python": "", # your python code.
"wolfram": "" # your Wolfram query.
}
Note: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\t'). 3. use 'print' function for the output.
4. Wait for me to give the results.
5. Continue to next step if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
Finally, when you get the answer, put the answer in \\boxed{}.
""",
    "v2refine": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step. Try to use python or wolfram to help you with one or more steps. Choose the best tool to be used.
Follow this format:
    - When query python, put code in ```python ... ```. Always use fractions instead of decimal and use the 'print' function for the output.
    - When query wolfram, put query ``wolfram ... ```
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get to the answer, please check the problem conditions to validate your answer. Correct yourself if necessary.
7. Finally, when you believe your answer is correct, put the answer in \\boxed{}.
""",
    # v1refine
    "v1refine": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.
First state the key idea to solve the problem. Then follow the process:
1. Solve the problem step by step. Try to use python or wolfram to help you with one or more steps. Put the query in json:
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Note: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct(use '\\t'). 3. use the 'print' function for the output.
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get to the answer, please check the problem conditions to validate your answer. Correct yourself if necessary.
7. Finally, when you believe your answer is correct, put the answer in \\boxed{}.
""",
    # v1nostep
    "v1nostep": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.

First state the key idea to solve the problem. Then follow the process:
1. Keep solving the problem and take out any queries that can be asked through python or Wolfram alpha.
Select the best tool and follow this format:
    - When query python. put code in ```python ... ```. Always use fractions instead of decimal and use the 'print' function for the output.
    - When query wolfram, put query ``wolfram ... ```
4. Wait for me to give the results.
5. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.
6. When you get the answer, put the answer in \\boxed{}.
""",
    # v3.2select bad change. 1. change case to mode, change mode 1, change mode 3
    "v3.2select": """Let's use two tools (python code and Wolfram alpha) to solve a math problem. Your are provided with three modes to solve the problem, choose the best mode to solve the problem and be flexible to switch between modes if necessary.

Query requirements:
You are provided with python code and Wolfram alpha, please choose the most suitable tool for each query.
You must put the query in json format (otherwise it will not be parsed correctly):
{"tool":"",# select the best tool from "python" or "wolfram".
"query":"" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1.always use fractions instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.

First state the key idea to solve the problem. You may choose from three modes to solve the problem:
Mode 1: If the problem is mostly reasoning or only involve simple calculations, you can solve it by yourself directly. After you get the answer, you can use tools to check your answer if necessary.
Mode 2: If the problem can be solved with python code directly, you can write a program to solve it. You should put the code in json following the query requirements. and I will help you run it.
Mode 3: If the problem cannot be handled with the above two modes, please follow this process:
1. Output one step (do not over divide the steps). Take out any queries that can be asked with the tools (for example, any calculations or equations that can be calculated) and follow the query requirements above.
2. Wait for me to give the results.
3. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning, or choose a different tool.

After all the queries are completed and you get the answer, put the answer in \\boxed{}.
""",
    # v0twostage
    "v0twostage": """Let's use two tools (python code and Wolfram alpha) to solve a math problem.

First state the key idea to solve the problem. Choose the best way from the two cases to solve the problem and be flexible to switch to another way if necessary.
Case 1: If the problem can be solved with python code directly, you can write a program to solve it.
Case 2: Otherwise, please solve it by yourself directly. You can use python code or Wolfram to help you when necessary (for calculations and equations, etc).

Whenenver you have a query, please follow the query requirements below. I will help you run the query and give you results.
Query requirements:
You are provided with python code and Wolfram alpha to help you, please choose the most suitable tool for each task.
You must put the query in json format (otherwise it will not be recognized):
{ "tool" : "", # select the best tool from "python" or "wolfram",
"query": "" # your query here, either python code or Wolfram query.
}
Caution: when you put python code in the query, you should: 1. try to use fractions/radical forms instead of decimal 2.make sure the indentation is correct (use '\\t'). 3. use the 'print' function for the output.

After all the queries are executed and you get the answer, put the answer in \\boxed{}.
""",
}
