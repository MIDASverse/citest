import numpy as np
import pandas as pd

from typing import Optional
from pydantic import BaseModel


class Dataset(BaseModel):
    """Object to store data and mask for missing data.

    This object stores, at minimum, the data with missing values and a mask of the missingness.

    For simulations, it can also store the full data.

    Fields:
        miss_data: A numpy array  with the data with missingness (recorded as np.nan)
        mask: A pandas DataFrame with the same shape as miss_data, with True where data is observed
        n: An integer with the number of observations in the data
        full_data: A pandas DataFrame with the full data (i.e. no missingness)
    """

    miss_data: pd.DataFrame = None
    mask: np.ndarray = None
    n: int = None
    full_data: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"""
            Dataset with {self.n} observations
            Outcome: {self.miss_data.columns[0]}
            
            {np.sum(~self.mask)} missing values
            """

    def make(self, data: pd.DataFrame, y=None, onehot=True):
        """Create a Dataset object from a pandas DataFrame to be used for the RL test.

        Args:
            data: A pandas DataFrame with missing values (recorded as np.nan)
            y: A string with the name of the outcome variable. If not provided,
                the first column will be assumed as the outcome.
            onehot: A boolean indicating whether to one-hot encode the data (default: True).
                Integer, float, and binary variables will not be encoded.

        """

        if self.miss_data is not None:
            raise ValueError(
                "Data already exists -- please create a new Dataset object"
            )

        if y is not None:
            data = pd.concat([data[y], data.drop(y, axis=1)], axis=1)

        data_wide = pd.get_dummies(data, dummy_na=True, dtype="boolean")

        for col in data.columns:
            if col + "_nan" in data_wide.columns:
                data_wide.loc[
                    data[col].isnull(), data_wide.columns.str.startswith(col + "_")
                ] = np.nan

                data_wide.drop(col + "_nan", axis=1, inplace=True)

        self.miss_data = data_wide
        self.mask = ~data_wide.isnull().to_numpy()
        self.n = data_wide.shape[0]
        self.full_data = None


def CIData(data: pd.DataFrame) -> Dataset:
    """Ingest missing data and format for the test

    This function takes in a pandas DataFrame with missing values, extracts
    a mask of the missingness, and returns a Dataset object.

    Args:
        data: A pandas DataFrame with missing values (recorded as np.nan)

    """

    pass


def v4_dgp(
    n: int,
    R_by: str,
    R_in: str,
) -> Dataset:
    """Generates a simple linear dataset with controllable missingness.

    Missing values can be set to be a function of either Y or X, and to be
    missing in those same columns too. This allows us to test every combination
    of MAR/MNAR and CI/NCI type of data.

    Args:
        n: Number of observations
        R_by: The column that determines missing values ('X' or 'Y')
        R_in: The column that will contain the missing values ('X' or 'Y')

    Returns:
        A Dataset object with the full data, missing data, and

    Raises:
        ValueError: An error generating or applying missing values to the chosen column
    """
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)
    X4 = np.random.normal(0, 1, n)
    X5 = np.random.normal(0, 1, n)
    Y = 5 * (X1 + X2 + X3 + X4 + X5) + np.random.normal(0, 1, n)

    full_data = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5})

    if R_by.upper() == "Y":
        R_latent = Y
    elif R_by.upper() == "X":
        R_latent = X1
    else:
        raise ValueError("R_by must be either 'Y' or 'X'")

    R = 1 * R_latent < np.quantile(R_latent, 0.5)

    if R_in.upper() == "X":
        X1[R == 0] = np.nan
    elif R_in.upper() == "Y":
        Y[R == 0] = np.nan
    else:
        raise ValueError("R_in must be either 'X' or 'Y'")

    corrupt_data = pd.DataFrame(
        {"Y": Y, "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5}
    )
    mask = ~corrupt_data.isnull().to_numpy()

    return Dataset(
        miss_data=corrupt_data,
        mask=mask,
        full_data=full_data,
        n=n,
    )


def MAR1(
    n: int,
    ci: bool = True,
) -> Dataset:
    """Generates the MAR-1 missing data pattern from King (2001).

    Args:
        n: Number of observations

    Returns:
        A Dataset object

    Raises:
        ValueError: An error generating or applying missing values to the chosen column
    """
    data = np.random.multivariate_normal(
        mean=[0, 0, 0, 0, 0],
        cov=[
            [1.0, -0.12, -0.1, 0.5, 0.1],
            [-0.12, 1.0, 0.1, -0.6, 0.1],
            [-0.1, 0.1, 1.0, -0.5, 0.1],
            [0.5, -0.6, -0.5, 1.0, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1],
        ],
        size=n,
    )

    M = np.ndarray(data.shape, dtype=bool)
    U1 = np.random.uniform(0, 1, n)

    # Y and X4 are MCAR:
    M[:, 0] = U1 < 0.85
    M[:, 4] = U1 < 0.85

    # X3 is always observed:
    M[:, 3] = True

    # X1 is MAR:
    U2 = np.random.uniform(0, 1, n)
    M[:, 1] = ~np.all([data[:, 3] < -1, U2 < 0.9], axis=0)

    # X2 is MAR but variable:
    U3 = np.random.uniform(0, 1, n)
    if ci:
        M[:, 2] = ~np.all(
            [data[:, 1] < -1, U3 < 0.9], axis=0
        )  # missing by X3 (original) but X1 means we can make it more complex as X1 also has missing values
    else:
        M[:, 2] = ~np.all([data[:, 0] < -1, U3 < 0.9], axis=0)  # missing by Y

    corrupt_data = data.copy()
    corrupt_data[~M] = np.nan

    return Dataset(
        miss_data=pd.DataFrame(corrupt_data),
        mask=M,
        full_data=pd.DataFrame(data),
        n=n,
    )
