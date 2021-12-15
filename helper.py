import pandas as pd
import numpy as np


def drop_subset(df: pd.DataFrame, col: str, cat: str) -> pd.DataFrame:
    """Takes a dataframe and removes a subset of the data.
    For example, removes shift-workers from a dataset.

    Paramters
    ---------
    df: pd.DataFrame
        Dataframe.
    col: str
        Name of the column to subset from.
    cat: str
        Name of the category to remove.

    Returns
    ------
    Copy of original dataframe with the category/subset removed.

    """

    data = df.copy()
    data = data[data[col].notna()]
    return data[~data[col].str.contains(cat)]


def groupby_describe(df: pd.DataFrame, group_on: str, to_describe: str) -> pd.DataFrame:
    """Takes a dataframe and returns summary statistics for a given
    feature based on a grouping of the data. For example, summary statistics
    per decile for height.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe.
    group_on: str
        Name of the column to group values by. For example, "Decile".
    to_describe: str
        Name of column to retrieve summary statistics for.

    Returns
    -------
    Dataframe with summary statistics for a particular column (to_describe),
    where each observation is a group from "group_on".
    """

    return df.groupby(group_on, observed=True)[to_describe].describe()


def bootstrap_stat(
    df: pd.DataFrame,
    col: str,
    sample_size: int,
    n_reps: int,
    low_conf: int,
    high_conf: int,
) -> tuple:
    """Takes  a dataframe, and column name, and computes a
    boostrapped confidence interval for 25, 50, 75 percentiles, returning the
    bootstrapped confidence intervals for each as a tuple where each element
     is an ndarray of the 95th confidence interval.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe.
    col: str
        Name of column to resample (with replacement) from.
    sample_size: int
        Sample size used in each calculation of the bootstrap.
    n_reps: int
        Number of repetitions (runs) for the boostrap.
    low_conf: int
        Lower bound of the confidence interval.
        For example 2.5 (low end of 95% conf)
    high_conf: int
        Upper bound of the confidence interval.
        For example 97.5 (high end of 95% conf)
    Returns
    -------
    A tuple of ndarrays, where each ndarray has two elements (the two bounds)
    of the confidence interval for each summary statistic.
    """

    # resample with replacement
    samples = np.random.choice(df[col], size=(n_reps, sample_size))

    # calculate summary statistics for each of the resamples
    twenty_fifth = np.percentile(samples, 25, axis=1)
    median = np.median(samples, axis=1)
    seventy_fifth = np.percentile(samples, 75, axis=1)

    # calculate confidence interval for each of the summary statistics
    twenty_fifth = np.around(
        np.percentile(twenty_fifth, [low_conf, high_conf]), decimals=2
    )
    median = np.around(np.percentile(median, [low_conf, high_conf]), decimals=2)
    seventy_fifth = np.around(
        np.percentile(seventy_fifth, [low_conf, high_conf]), decimals=2
    )

    return (twenty_fifth, median, seventy_fifth)


def bootstrap(
    df: pd.DataFrame,
    col: str,
    sample_size: int = 100,
    n_reps: int = 10000,
    low_conf: int = 2.5,
    high_conf: int = 97.5,
) -> pd.DataFrame:
    """Takes a dataframe, and for each decile (10 year age group), calculates
    a confidence interval for summary statistics for each decile.
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe.
    col: str
        Name of column to resample (with replacement) from.
    sample_size: int
        Sample size used in each calculation of the bootstrap.
    n_reps: int
        Number of repetitions (runs) for the boostrap.
    low_conf: int
        Lower bound of the confidence interval.
        For example 2.5 (low end of 95% conf)
    high_conf: int
        Upper bound of the confidence interval.
        For example 97.5 (high end of 95% conf)
    Returns
    -------
    A dataframe with the of the confidence intervals for the summary stats
    for each decile.
    """

    # create storage for each of the confidence intervals (low and high bounds)
    twenty_fifth_low = np.empty(0)
    twenty_fifth_high = np.empty(0)

    median_low = np.empty(0)
    median_high = np.empty(0)

    seventy_fifth_low = np.empty(0)
    seventy_fifth_high = np.empty(0)

    # get each of the deciles
    deciles = df.groupby("Decile", observed=True).count().index.to_list()

    for decile in deciles:
        # bootstrap for each of the deciles
        twenty_fifth_conf, median_conf, seventy_fifth_conf = bootstrap_stat(
            df[df["Decile"] == decile], col, sample_size, n_reps, low_conf, high_conf
        )

        # append summary stats for each of the deciles
        twenty_fifth_low = np.append(twenty_fifth_low, twenty_fifth_conf[0])
        twenty_fifth_high = np.append(twenty_fifth_high, twenty_fifth_conf[1])

        median_low = np.append(median_low, median_conf[0])
        median_high = np.append(median_high, median_conf[1])

        seventy_fifth_low = np.append(seventy_fifth_low, seventy_fifth_conf[0])
        seventy_fifth_high = np.append(seventy_fifth_high, seventy_fifth_conf[1])

    return pd.DataFrame(
        {
            "Decile": deciles,
            "25th low": twenty_fifth_low,
            "25th high": twenty_fifth_high,
            "50th low": median_low,
            "50th high": median_high,
            "75th low": seventy_fifth_low,
            "75th high": seventy_fifth_high,
        }
    ).set_index("Decile")
