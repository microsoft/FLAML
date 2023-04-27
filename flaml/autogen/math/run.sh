


# python main.py --prompt_type v1select --prompt_location user --categories 0
# python main.py --prompt_type v1select --prompt_location system --categories 0
# python main.py --prompt_type nostep --prompt_location user --categories 0
# python main.py --prompt_type nostep --prompt_location system --categories 0


python main.py --prompt_type v2select --prompt_location user --categories 0
python main.py --prompt_type v1nostep --prompt_location user --categories 0

python main.py --prompt_type v2select --prompt_location user --categories 1
python main.py --prompt_type v1nostep --prompt_location user --categories 1

python main.py --prompt_type v1refine --prompt_location user --categories 0
python main.py --prompt_type v2refine --prompt_location user --categories 0