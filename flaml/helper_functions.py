import numpy as np


def format_integers(integer: float) -> str:
    """
    Format a floating-point number to a string with a dynamic number of significant figures.

    The number of significant figures is determined based on the position of the first
    non-zero digit after the decimal point, with a minimum of 4 significant figures.

    Parameters:
    - integer (float): The floating-point number to format.

    Returns:
    - str: The formatted number as a string.

    Example:
    - format_integers(0.0003123) returns '0.0003123'
    - format_integers(0) returns '0.0000'
    """
    if integer == 0:
        return "0.0000"  # Return a standard format for zero to maintain four decimal places.

    # Determine the first non-zero digit's position after the decimal
    position = next((i for i, c in enumerate(str(integer).split(".")[-1]) if c.isdigit() and c != "0"), -1)

    # Apply formatting: Use a minimum of 4 significant figures if the first significant digit occurs early
    return "{:.{}g}".format(integer, position if position > 4 else 4)
