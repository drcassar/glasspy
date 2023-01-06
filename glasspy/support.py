import numpy as np
from functools import reduce


def round_to_closest_base(x, base):
    """Round numbers to closest base number.

    Args:
      x:
        Array-like object with numbers.
      base:
        Float to which the numbers will be rounded.

    Returns:
      Numpy array with the rounded numbers.

    Note:
      Implementation by Alok Singhal found in
      https://stackoverflow.com/a/2272174.
    """
    return base * np.round(x / base)


def round_to_significant_figure(x, n):
    """Round numbers to n sifnificant digits.

    Args:
      x:
        Array-like object with numbers.
      n:
        Integer representing the number of significant digits to round to. Must
        be the same shape as `x`.

    Returns:
      Numpy array with the numbers rounded to the desired significant digits.

    Note:
      Implementation by Scott Gigante found in
      https://stackoverflow.com/a/59888924.
    """

    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (n - 1))
    mags = 10 ** (n - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def round_number_with_error(x, error):
    """Round numbers with respect of their errors.

    x:
      Array-like object with numbers. These are the values that will be rounded.
    error:
      Array-like object with numbers. These are the errors that will guid the
      rounding of x.

    Returns:
      x:
        Rounded values of x.
      error:
        Error of x with 1 significant digit.
      text:
        String array with the values of x with error represented in parenthesis.

    Raises:
      AssertionError:
        Happens when you have a negative value in the `error` argument. Errors
        must be positive.
    """

    error = np.asarray(error)
    x = np.asarray(x)

    assert (error >= 0).all(), "Error values must be positive."

    error = round_to_significant_figure(error, 1)
    digit_location = np.floor(np.log10(error)).astype(int)

    x = np.array(
        [np.round(v, -p) for v, p in zip(x.ravel(), digit_location.ravel())]
    ).reshape(x.shape)

    formatting = np.array(
        [f"%.{-d}f" if d < 1 else f"%.0f" for d in digit_location.ravel()]
    ).reshape(x.shape)
    value_text = np.char.mod(formatting, x)

    error_digit = np.round(error * 10.0 ** (-digit_location), 0).astype(int)
    error_text = np.where(
        digit_location < 1,
        error_digit.astype(str),
        (error_digit * 10 ** np.abs(digit_location)).astype(int).astype(str),
    )

    text = reduce(np.char.add, [value_text, "(", error_text, ")"])

    return x, error, text
