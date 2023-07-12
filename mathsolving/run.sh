
# test level-5 problems
python main.py --solver mathchat --categories 0 1 3 4 5 6 --samples_per_category 10
python main.py --solver func_python --categories 0 1 3 4 5 6 --samples_per_category 10
python main.py --solver func_wolfram --categories 0 1 3 4 5 6 --samples_per_category 10

# test random sampled problems
python main.py --solver mathchat --sample_all 100
python main.py --solver func_python --sample_all 100
python main.py --solver func_wolfram --sample_all 100
