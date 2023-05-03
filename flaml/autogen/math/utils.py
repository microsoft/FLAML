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
            print(message, flush=True)


def load_fixed(category_to_load=None):
    category_to_load = [i for i in range(7)] if not category_to_load or "all" in category_to_load else category_to_load
    category_to_load = [int(x) for x in category_to_load]
    folder = "22_user_v3select_t1"
    sep_cat = []

    for i, category in enumerate(math_type_mapping.keys()):
        if i not in category_to_load:
            continue

        c = math_type_mapping[category]
        sep_cat.append([])
        for i in range(20):
            with open(os.path.join(folder, c, f"{i}.json"), "r") as fp:
                problem = json.load(fp)
            del problem["is_valid_reply"]
            del problem["is_correct"]
            del problem["correct_ans"]
            del problem["voted_answer"]
            del problem["round"]
            del problem["valid_q_count"]
            del problem["total_q_count"]
            del problem["cost"]
            del problem["messages"]

            sep_cat[-1].append(problem)
    return sep_cat


def load_level5_math_each_category(samples_per_category=20, category_to_load=None):
    """
    Load level 5 math problems from the competition dataset.
    Returns:
        A list of list of problems. Each list of problems is of the same category.
    """
    category_to_load = [i for i in range(7)] if not category_to_load or "all" in category_to_load else category_to_load
    category_to_load = [int(x) for x in category_to_load]
    seed = 41
    data = datasets.load_dataset("competition_math")
    test_data = data["test"].shuffle(seed=seed)
    sep_cate = []
    for i, category in enumerate(math_type_mapping.keys()):
        if i not in category_to_load:
            print(i, category, "(skipped)", flush=True)
            continue
        print(i, category, flush=True)
        tmp = [
            test_data[x]
            for x in range(len(test_data))
            if test_data[x]["level"] == "Level 5" and test_data[x]["type"] == category
        ]
        if len(tmp) < samples_per_category:
            print(f"Warning: {category} has less than {samples_per_category} problems.", flush=True)
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
