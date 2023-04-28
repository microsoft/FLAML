


# python main.py --prompt_type v1select --prompt_location user --categories 0
# python main.py --prompt_type v1select --prompt_location system --categories 0
# python main.py --prompt_type nostep --prompt_location user --categories 0
# python main.py --prompt_type nostep --prompt_location system --categories 0


# trial 1 v1.1select user
python main.py -ptype v1.1select --prompt_location user --folder ./1 --categories 0 1

# trial 2 v1.2select user
python main.py -ptype v1.2select --prompt_location user --folder ./2 --categories 0 1

# trial 3 v2.1select user
python main.py -ptype v2.1select --prompt_location user --folder ./3 --categories 0 1

# trial 4 v1.1select system
python main.py -ptype v1.1select --prompt_location system --folder ./4 --categories 0 1

# trial 5 v1.2select system
python main.py -ptype v1.2select --prompt_location system --folder ./5 --categories 0 1

# trial 6 v2.1select system
python main.py -ptype v2.1select --prompt_location system --folder ./6 --categories 0 1

# ------------------------------
# trial 7 v2.1select user continued
python main.py -ptype v2.1select --prompt_location user --folder ./7 --categories 0 1 4 5

# trial 8 v1.1select system
python main.py -ptype v1.2select --prompt_location system --folder ./8 --categories 0 1 4 5

# trial 9 v3select experimental user
python main.py -ptype v3select --prompt_location user --folder ./9 --categories 0 1

# trial 10 v3select experimental system
python main.py -ptype v3select --prompt_location system --folder ./10 --categories 0 1
