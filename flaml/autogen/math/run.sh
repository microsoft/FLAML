


# python main.py --prompt_type v1select --prompt_location user --categories 0
# python main.py --prompt_type v1select --prompt_location system --categories 0
# python main.py --prompt_type nostep --prompt_location user --categories 0
# python main.py --prompt_type nostep --prompt_location system --categories 0


# trial 1 v1.1select user
# python main.py -ptype v1.1select --prompt_location user --folder ./1 --categories 0 1

# # trial 2 v1.2select user
# python main.py -ptype v1.2select --prompt_location user --folder ./2 --categories 0 1

# # trial 3 v2.1select user
# python main.py -ptype v2.1select --prompt_location user --folder ./3 --categories 0 1

# # trial 4 v1.1select system
# python main.py -ptype v1.1select --prompt_location system --folder ./4 --categories 0 1

# # trial 5 v1.2select system
# python main.py -ptype v1.2select --prompt_location system --folder ./5 --categories 0 1

# # trial 6 v2.1select system
# python main.py -ptype v2.1select --prompt_location system --folder ./6 --categories 0 1

# ------------------------------
# trial 7 v2.1select user continued
# python main.py -ptype v2.1select --prompt_location user --folder ./7 --categories 0 1 4 5

# trial 8 v1.1select system
# python main.py -ptype v1.2select --prompt_location system --folder ./8 --categories 0 1 4 5

# trial 9 v3select user
# python main.py -ptype v3select --prompt_location user --folder ./9 --categories 0 1 4 5

# trial 10 v3select experimental system
# python main.py -ptype v3select --prompt_location system --folder ./10 --categories 0 1

# trial 11 v3.1select user
# python main.py -ptype v3.1select --prompt_location user --folder ./11 --categories 0 1

# trial 12 v3.1select system
# python main.py -ptype v3.1select --prompt_location system --folder ./12 --categories 0 1

# trial 15 v1both user
# python main.py -ptype v1both --prompt_location user --folder ./15 --categories 0 1 5

# # trial 13 v3.2select user
# python main.py -ptype v3.2select --prompt_location user --folder ./13 --categories 0 1 4 5

# # trial 14 v3.2select system
# python main.py -ptype v3.2select --prompt_location system --folder ./14 --categories 0 1 4 5


# # trial 16 v3select user
# python main.py -ptype v3select --prompt_location user --folder ./16 --categories 0 1 4 5

# # trial 17 v3select user temp=0
# python main.py -ptype v3select --prompt_location user --temperature 0 --folder ./17 --categories 0 1 4 5

# # trial 18 v3.3select user
# python main.py -ptype v3.3select --prompt_location user --folder ./18 --categories 0 1 4 5


# ------------------------------Switching to new set of samples------------------------------
# # trial 19 baseline zeroshot
# python baselines/zeroshot.py --folder ./19

# # trial 20 v3python user
# python main.py -ptype v3python --prompt_location user --folder ./20 --categories 0 1 4 5

# # trial 21 select user run on all categories
# python main.py -ptype select --prompt_location user --folder ./21 --categories all

# trial 22 v3select user
# python main.py -ptype v3select --prompt_location user --folder ./22 --categories all

# trial 23 PoT
# python baselines/PoT.py --folder ./23


# trial 24 PoT baseline another seed for cache
# python baselines/PoT.py --folder ./24 --seed 42

# trial 25 v1.3select user
# difference from original select: 1: new query format
# desired observation: 1. less errors 3. similar or better performance
# python main.py -ptype v1.3select --prompt_location user --folder ./25 --categories 0 4

# trial 26 v3.4select user
# analyse runs from v3.1 and v3.2 and aggregate prompts that are related to desired behaviour
# python main.py -ptype v3.4select --prompt_location user --folder ./26 --categories 0 1 4 5

# trial 27 v1.4select user
# difference from original select: 1: new query format. 2: add "do not overdivide step"
# desired observation: 1. less errors, 2. less rounds 3. similar or better performance
# python main.py -ptype v1.4select --prompt_location user --folder ./27 --categories 0 1 4
# todo: add note for wolfram to compare if trial 24 has desired effect

# trial 28 v1.5
# from v1.4
# suppose to be 28 but the name is 27
# python main.py -ptype v1.5select --prompt_location user --folder ./27 --categories 0 1 4 5 --select


# switch to code block format for queries
# trial 29 v3.1python: on top of v3python, explicitly adding "using loops to enumerate all possible cases"
# tested correct on counting 0, 1, 9 which was wrong in v3python
python main.py -ptype v3.1python --folder ./29 --categories 0 1 4 5

# switch to code block format for queries
# trial 30 v3.5select: on top of v3python, adding wolfram and nothing else
# can easily solve algebra 6, 9 that v3python cannot hardly get correct
python main.py -ptype v3.5select --folder ./30 --categories 0 1 4 5
