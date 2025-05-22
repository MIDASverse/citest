import numpy as np
import pandas as pd

from typing import Optional
from pydantic import BaseModel

from importlib.resources import files
from re import compile


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

    @staticmethod
    def _dummy(data: pd.DataFrame, drop_first=False) -> pd.DataFrame:
        """Create a one-hot encoded version of the data.

        Args:
            data: A pandas DataFrame with missing values (recorded as np.nan)
            drop_first: A boolean indicating whether to drop the first level of each variable

        """
        data_wide = pd.get_dummies(
            data, dummy_na=True, dtype="boolean", drop_first=drop_first
        )

        for col in data.columns:
            if col + "_nan" in data_wide.columns:
                data_wide.loc[
                    data[col].isnull(), data_wide.columns.str.startswith(col + "_")
                ] = np.nan

                data_wide.drop(col + "_nan", axis=1, inplace=True)

        return data_wide

    def make(self, data: pd.DataFrame, y=None, _onehot=True):
        """Create a Dataset object from a pandas DataFrame to be used for the RL test.

        Args:
            data: A pandas DataFrame with missing values (recorded as np.nan)
            y: A string with the name of the outcome variable. If not provided,
                the first column will be assumed as the outcome.
            _onehot: A boolean indicating whether to one-hot encode the data (default: True).
                Integer, float, and binary variables will not be encoded.

        """

        if self.miss_data is not None:
            raise ValueError(
                "Data already exists -- please create a new Dataset object"
            )

        if y is not None:
            data = pd.concat([data[y], data.drop(y, axis=1)], axis=1)

        data_wide = self._dummy(data) if _onehot else data

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


def adult(n=1000, ci=True, mcar_prop=0.5) -> Dataset:

    path = files("citest.data_examples").joinpath("us-census-income.csv")
    adult = pd.read_csv(path)

    idxs = np.random.choice(adult.shape[0], n)
    adult_compl = adult.iloc[idxs, :]

    adult_compl.loc[:, "income"] = adult_compl["income"].map({"<=50K": 0, ">50K": 1})

    adult_compl = pd.concat(
        [adult_compl["income"], adult_compl.drop("income", axis=1)], axis=1
    ).reset_index(drop=True)

    adult_compl["income"] = adult_compl["income"].astype("int64")

    a_dataset = Dataset()
    adult_wide = a_dataset._dummy(adult_compl)

    ed_cols = list(filter(compile("^education_").match, adult_wide.columns.tolist()))

    # Missing pattern
    adult_miss = adult_wide.copy()

    if not ci:
        for i in range(adult_miss.shape[0]):
            if adult_miss["income"].iloc[i] == 1 and np.random.rand() < 0.9:
                adult_miss.loc[i, ed_cols] = pd.NA

    else:
        for i in range(adult_miss.shape[0]):
            if adult_miss["age"].iloc[i] <= 30 and np.random.rand() < 0.9:
                adult_miss.loc[i, ed_cols] = pd.NA

    for c in np.random.choice(
        a=adult_miss.shape[1] - 1,
        size=int(adult_miss.shape[1] * mcar_prop),
    ):
        adult_miss.iloc[
            np.random.choice(adult_miss.shape[0], int(adult_miss.shape[0] * 0.5)), c + 1
        ] = np.nan

    # Make dataset object
    a_dataset.make(adult_miss, y="income")
    a_dataset.full_data = a_dataset._dummy(adult_compl)

    assert a_dataset.full_data.shape == a_dataset.miss_data.shape
    assert (a_dataset.full_data.columns == a_dataset.miss_data.columns).all()

    return a_dataset


def mushrooms(n=1000, ci=True, mcar_prop=0.5) -> Dataset:

    path = files("citest.data_examples").joinpath("agaricus-lepiota.data")
    mushrooms = pd.read_csv(path, delimiter=",", header=None)
    mushrooms.columns = ["y"] + [f"X{i}" for i in range(1, mushrooms.shape[1])]
    idxs = np.random.choice(mushrooms.shape[0], n)

    mushrooms_compl = mushrooms.iloc[idxs, :].copy()

    mushrooms_compl.loc[:, "y"] = mushrooms_compl["y"].map({"p": 0, "e": 1})

    assert (
        mushrooms_compl.columns[0] == "y" and mushrooms_compl.iloc[:, 0].nunique() == 2
    )

    mushrooms_compl["y"] = mushrooms_compl["y"].astype("int64")

    # Make dataset object
    m_dataset = Dataset()
    m_wide = m_dataset._dummy(mushrooms_compl)

    odor_cols = list(filter(compile("^X5_").match, m_wide.columns.tolist()))

    # Missing pattern
    mushrooms_miss = m_wide.copy()

    if not ci:
        for i in range(mushrooms_miss.shape[0]):
            if mushrooms_miss["y"].iloc[i] == 1 and np.random.rand() < 0.9:
                mushrooms_miss.loc[i, odor_cols] = pd.NA

    else:
        for i in range(mushrooms_miss.shape[0]):
            if mushrooms_compl["X6"].iloc[i] == "t" and np.random.rand() < 0.9:
                mushrooms_miss.loc[i, odor_cols] = pd.NA

    for c in np.random.choice(
        a=mushrooms_miss.shape[1] - 1,
        size=int(mushrooms_miss.shape[1] * mcar_prop),
    ):
        mushrooms_miss.iloc[
            np.random.choice(
                mushrooms_miss.shape[0], int(mushrooms_miss.shape[0] * 0.5)
            ),
            c + 1,
        ] = np.nan

    # Make dataset object
    m_dataset.make(mushrooms_miss, y="y")
    m_dataset.full_data = m_dataset._dummy(mushrooms_compl)

    assert m_dataset.full_data.shape == m_dataset.miss_data.shape
    assert (m_dataset.full_data.columns == m_dataset.miss_data.columns).all()

    return m_dataset
