


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
# python main.py -ptype v3.1python --folder ./29 --categories 0 1 4 5

# switch to code block format for queries
# trial 30 v3.5select: on top of v3python, adding wolfram and nothing else
# can easily solve algebra 6, 9 that v3python cannot hardly get correct
# python main.py -ptype v3.5select --folder ./30 --categories 0 1 4 5


# trial 31 v3.6select  v3.1python+wolfram, especially remove "Wolfram might be suitable for symbolic manipulations"
# python main.py -ptype v3.6select --folder ./31 --categories 0

# trial 32 v3.7select  v3.6select refine
# 1. Change to “# your single wolfram query”
# 2. Add:  I will help you execute queries.
# 3. Add: (the solving process can be written with code)
# 4. Add: Do not complicate the problems.
# 5. wording: “should always” -> must
# 6. wording: “should” -> must notes in query requirements
# 7. Remove: (otherwise it will not be recognized)
# python main.py -ptype v3.7select --folder ./32 --categories 0 --select



# v3.7select. Hard to say adding wolfram is good or bad. So test on other categories
# python main.py -ptype v3.7select --folder ./32 --categories 0 1 4 5

# trial 33: v3.1python again with refine
# python main.py -ptype v3.1python --folder ./33 --categories 0 1 4 5 --refine

# trial 34: v3.2python, slightly refine v3.1python and some refinements in query handling
# should have very similar performance to v3.1python
# python main.py -ptype v3.2python --folder ./34 --categories 0 1 4 5 --refine

# trial 35: v3.3python, based on previous best v1select, this is a test
# python main.py -ptype v3.3python --folder ./35 --categories 0 1

# -------------------test on random sampled problems from the whole dataset-------------------
# trial 36: PoT on random sampled problems
# python PoT.py --folder ./36 --sample_all 100

# # trial 37: v3.1python on random sampled problems
# python main.py -ptype v3.1python --folder ./37 --sample_all 100


# trial 38: v3.1 test on 50 level 5 problems per category
# python main.py -ptype v3.1python --folder ./38 --categories 0 1 3 4 5 6 --samples_per_category 50

# trial 39: v3.1 with a different system message
# python main.py -ptype v3.1python --folder ./39s1 --sample_all 100

# # trial 40: v3.2
# python main.py -ptype v3.2python --folder ./40s1 --sample_all 100

# # trial 41: v4.2
# python main.py -ptype v4.2python --folder ./41s1 --sample_all 100

# # trial 42: v3.1 with a different system message on 20 level5 problems
# python main.py -ptype v3.1python --folder ./42s1 --categories 0 1 4 5


# -------------------test on 50 level 5 problems per category for 6 categories-------------------
# trial 43: PoT on 50 level 5 problems per category
# python PoT.py --folder ./43 --categories 0 1 3 4 5 6 --samples_per_category 50

# # trial 47: v4.2 with a different system message on 20 level5 problems
# python pnas.py --folder ./47 --sample_all 100

# # trial 48: v4.2 with a different system message on 20 level5 problems
# python main.py -ptype v4.2python --folder ./48 --categories 0 1 4 5

# # trial 49: v4.2 with s2 system
# python main.py -ptype v3.1python -systype s2 --folder ./49 --categories 0 1 4 5

# # trial 50: v4.2 with s2 system
# python main.py -ptype v4.2python -systype s2 --folder ./50 --categories 0 1 4 5


# # # trial 51: v3.1 remove specific message
# python main.py -ptype v3.2python --folder ./51 --categories 0 1 3 4 5 6

# # # trial 52: v4.2  original system message
# # python main.py -ptype v4.2python --folder ./52 --categories 0 1 3 4 5 6

# # # trial 53: v1.4  original system message
# python main.py -ptype v1.6select --folder ./53 --categories 0 1 4 5


# # trial 44: zeroshot on 50 level 5 problems per category
# # python zeroshot.py --folder ./44 --categories 0 1 3 4 5 6 --samples_per_category 50

# # trial 45: PoT on all level-5 problems from 6 categories
# python PoT.py --folder ./45 --categories 0 1 3 4 5 6  --samples_per_category 400

# # trial 46: zeroshot on all level-5 problems from 6 categories
# python zeroshot.py --folder ./46 --categories 0 1 3 4 5 6 --samples_per_category 400



# trial 56: general_1, finalized from v3python
# python main.py -ptype general_1 --folder ./56 --categories 0 1 4 5

# # trial 59: specific_1, finalized from v3.1python
# python main.py -ptype specific_1 --folder ./59 --categories 0 1 4 5

# trial 55: v3.2  original system message, finish running v3.2
# python main.py -ptype v3.3python --folder ./55 --categories 0 1

# # trial 57: general_2, finalized from v3python
# python main.py -ptype general_2 --folder ./57 --categories 0 1

# # trial 58: general_3, change from v3.1python
# python main.py -ptype general_3 --folder ./58 --categories 0 1



# trial 60: v3.1python
# python main.py -ptype v3.1python --folder ./60 --categories 3 4 5 6

# # trial 61: general_4
# python main.py -ptype general_4 --folder ./61 --categories 0 1


# # trial 62: general_5
# python main.py -ptype general_5 --folder ./62 --categories 0 1

# python main.py -ptype general_1 --folder ./56 --select

# python main.py -ptype v3.5python --folder ./63 --select


# trial 55: v3.2  original system message, finish running v3.2
# python main.py -ptype v3.3python --folder ./55 --categories 0 1 3 4 5 6 --samples_per_category 50


# trial 63: v3.6python
# python main.py -ptype v3.6python --folder ./63 --categories 0

# # trial 64: v3.7python
# python main.py -ptype v3.7python --folder ./64 --categories 0

# # trial 65: v3.8python
# python main.py -ptype v3.8python --folder ./65 --categories 0


# python main.py -ptype v3.9python --folder ./66 --select

# trial 67:
# python main.py -ptype v3.9python --folder ./67 --categories 0 1 3 4 5 6 --samples_per_category 50

# # trial 68: zero-shot on 50
# python zeroshot.py --folder ./68 --categories 0 1 3 4 5 6 --samples_per_category 50

# trial 68: zero-shot on 50
# python zeroshot.py --folder ./68 --categories 0 1 3 4 5 6 --samples_per_category 400

# # trial 67: continue run ours on all the problems
# python main.py -ptype v3.9python --folder ./67 --categories 0 1 3 4 5 6 --samples_per_category 400

# # trial 69: run pnas on 50 problems from each category
# python pnas.py --folder ./69 --samples_per_category 50 --categories 0 1 3 4 5 6

# # trial 69: run pnas on all problems from each category
# python pnas.py --folder ./69 --samples_per_category 400 --categories 0 1 3 4 5 6


# trial 70: run pnas on all problems from each category
# python main.py -ptype v1python --folder ./70 --samples_per_category 50 --categories 0 1 3 4 5 6

# trial 70: run pnas on all problems from each category
# python main.py -ptype v1python --folder ./70 --samples_per_category 400 --categories 0 1 3 4 5 6

# trial 71: run v1 python + wolfram
# python main.py -ptype v1final_select --folder ./71 --samples_per_category 50 --categories 0 1 3 4 5 6

# trial 72
python PoT.py --folder ./72 --samples_per_category 50 --categories 0 1 3 4 5 6
