


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

# # trial 21 select user
# python main.py -ptype select --prompt_location user --folder ./21 --categories 0 1 4 5


# trial 22 v3select user
python main.py -ptype v3select --prompt_location user --folder ./22 --categories all

# trial 23 PoT
python baselines/PoT.py --folder ./23