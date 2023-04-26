import datasets
import re
import os
import json
import argparse

math_type_mapping = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Math Solver")
    parser.add_argument("--prompt_type", dest="prompt_type", help="prompt type", default="select", type=str)
    parser.add_argument("--prompt_location", dest="prompt_location", help="prompt location", default="user", type=str)
    parser.add_argument("--max_round", dest="max_round", help="max round", default=15, type=int)
    parser.add_argument("--folder", "-f", dest="folder", help="saving folder", default="./autotools", type=str)
    parser.add_argument("--cache_folder", "-c", dest="cache_folder", default=".cache", help="cache folder")
    parser.add_argument("--samples_per_category", help="samples per category", default=20, type=int)
    parser.add_argument("--temperature", "-t", dest="temperature", help="temperature", default=1, type=float)
    parser.add_argument("--test_run", help="test run", action="store_true")
    parser.add_argument("--categories", dest="categories", help="categories", default=[0, 1], nargs="+")

    # not used

    parser.add_argument("--n", dest="n", help="number of samples", default=1, type=int)
    parser.add_argument("--voting", action="store_true")
    args = parser.parse_args()
    args.folder = args.folder + "_" + args.prompt_location + "_" + args.prompt_type + "_t" + str(args.temperature)
    os.makedirs(args.folder, exist_ok=True)
    return args


class mylogger:
    def __init__(self, file) -> None:
        self.file = file

    def log(self, message, verbose=True):
        """Print the message.
        Args:
            message (str): The message to print.
        """
        with open(self.file, "a") as f:
            f.write(message + "\n")
        if verbose:
            print(message)


def load_level5_math_each_category(samples_per_category=20, category_to_load=None):
    """
    Load level 5 math problems from the competition dataset.
    Returns:
        A list of list of problems. Each list of problems is of the same category.
    """
    category_to_load = [i for i in range(7)] if not category_to_load or "all" in category_to_load else category_to_load
    seed = 41
    data = datasets.load_dataset("competition_math")
    test_data = data["test"].shuffle(seed=seed)
    sep_cate = []
    for i, category in enumerate(math_type_mapping.keys()):
        if i not in category_to_load:
            print(i, category, "(skipped)")
            continue
        print(i, category)
        tmp = [
            test_data[x]
            for x in range(len(test_data))
            if test_data[x]["level"] == "Level 5" and test_data[x]["type"] == category
        ]
        if len(tmp) < samples_per_category:
            print(f"Warning: {category} has less than {samples_per_category} problems.")
        sep_cate.append(tmp[:samples_per_category])

    if len(sep_cate) == 0:
        raise ValueError("No category is loaded.")
    return sep_cate


def remove_asy_sections(input_string):
    """Remove asy sections from the input string.

    Args:
        input_string (str): The input string.
    Returns:
        str: The string without asy sections.
    """
    pattern = r"\[asy\](.*?)\[\\asy\]"
    output_string = re.sub(pattern, "", input_string, flags=re.DOTALL)
    pattern = r"\[asy\](.*?)\[/asy\]"
    output_string = re.sub(pattern, "", output_string, flags=re.DOTALL)
    pattern = r"\[ASY\](.*?)\[\\ASY\]"
    output_string = re.sub(pattern, "", output_string, flags=re.DOTALL)
    pattern = r"\[ASY\](.*?)\[/ASY\]"
    output_string = re.sub(pattern, "", output_string, flags=re.DOTALL)
    return output_string


def write_json(dict_to_save, file):
    """Write a dictionary to a json file.
    Args:

        dict_to_save (dict): The dictionary to save.
        file (str): The file to save to.
    """
    jstring = json.dumps(dict_to_save, indent=2)
    with open(file, "w") as j:
        j.write(jstring)
