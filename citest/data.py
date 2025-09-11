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
        expl_vars: A list of column names specifying subset of columns used in analysis models (typically set using the `make` method).
    """

    miss_data: pd.DataFrame = None
    mask: np.ndarray = None
    n: int = None
    full_data: Optional[pd.DataFrame] = None
    expl_vars: Optional[list] = None

    # private vars:
    _expl_vars: list = (
        []
    )  # private variable that will store correctly formatted one-hot encoded vars too

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):

        if self.miss_data is None:
            return "Dataset not fit to data yet. Please use the `make` method to create a Dataset object."
        else:
            return f"""
                Dataset with {self.n} observations
                Outcome: {self.miss_data.columns[0]}
                Explanatory variables: {self.expl_vars}
                
                {round(100*np.sum(~self.mask)/np.prod(self.mask.shape),1)}% missing values
                """

    def _dummy(self, data: pd.DataFrame, drop_first=False) -> pd.DataFrame:
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

                exp_cols = data_wide.columns.str.startswith(col + "_")

                if col in self.expl_vars:
                    self._expl_vars.extend(
                        data_wide.columns[
                            exp_cols
                            & ~data_wide.columns.str.endswith(
                                "_nan"
                            )  # drop na indicator from tracker
                        ]
                    )

                data_wide.loc[data[col].isnull(), exp_cols] = np.nan

                data_wide.drop(col + "_nan", axis=1, inplace=True)

            elif col in self.expl_vars:
                self._expl_vars.append(col)

        return data_wide

    def make(self, data: pd.DataFrame, y: str, expl_vars=None, _onehot=True):
        """Create a Dataset object from a pandas DataFrame to be used for the RL test.

        Args:
            data: A pandas DataFrame with missing values (recorded as np.nan)
            y: A string with the name of the outcome variable. If not provided,
                the first column will be assumed as the outcome.
            expl_vars: A list of strings with the names of variables to be included in the conditional analysis. If not provided,
                all columns except the outcome will be used. Note, more variables can be  used in the imputation stage.
            _onehot: A boolean indicating whether to one-hot encode the data (default: True).
                Integer, float, and binary variables will not be encoded.

        """

        if self.miss_data is not None:
            raise ValueError(
                "Data already exists -- please create a new Dataset object"
            )

        if y in data.columns:
            data = pd.concat([data[y], data.drop(y, axis=1)], axis=1)
        else:
            raise ValueError(
                "Outcome variable not found in data. Please provide a valid outcome variable name."
            )

        if expl_vars is not None:
            self.expl_vars = expl_vars
        else:
            self.expl_vars = data.columns.tolist()[1:]

        if _onehot:
            data_wide = self._dummy(data)
        else:
            data_wide = data.copy()
            self._expl_vars = self.expl_vars.copy()

        self.miss_data = data_wide
        self.mask = ~data_wide.isnull().to_numpy()
        self.n = data_wide.shape[0]
        self.full_data = None

        self._expl_vars = data_wide.columns.get_indexer(self._expl_vars).tolist()


def kuha(
    n: int,
    R_by: str,
    R_in: str,
    inc_Z: bool = False,
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
    X = np.random.normal(0, 1, n)
    Z = np.random.normal(0, 1, n)

    if inc_Z:
        Y = 5 * X + np.random.normal(0, 1, n)
        full_data = pd.DataFrame({"Y": Y, "X": X, "Z": Z})
    else:
        Y = 5 * X + np.random.normal(0, 1, n)
        full_data = pd.DataFrame({"Y": Y, "X": X})

    if R_by.upper() == "Y":
        R_latent = Y
    elif R_by.upper() == "X":
        R_latent = X
    else:
        raise ValueError("R_by must be either 'Y' or 'X'")

    R = 1 * (R_latent < np.quantile(R_latent, 0.5))

    if R_in.upper() == "X":
        X[R == 0] = np.nan
    elif R_in.upper() == "Y":
        Y[R == 0] = np.nan
    else:
        raise ValueError("R_in must be either 'X' or 'Y'")

    if inc_Z:
        corrupt_data = pd.DataFrame({"Y": Y, "X": X, "Z": Z})
    else:
        corrupt_data = pd.DataFrame({"Y": Y, "X": X})

    kuha_dataset = Dataset()
    kuha_dataset.make(corrupt_data, y="Y")
    kuha_dataset.full_data = pd.DataFrame(full_data)
    return kuha_dataset


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

    v4_dataset = Dataset()
    v4_dataset.make(corrupt_data, y="Y")
    v4_dataset.full_data = pd.DataFrame(full_data)
    return v4_dataset


def identify(
    n: int,
    ci: str,
    eta: float = 0.0,
) -> Dataset:

    Z = np.random.normal(loc=0, scale=1, size=n)
    X = np.random.normal(loc=eta * Z, scale=1, size=n)
    Y = np.random.normal(loc=0.5 * X + 0.5 * Z, scale=1, size=n)

    if ci:
        R = np.random.binomial(1, 1 / (1 + np.exp(-X)), size=n)
    else:
        R = np.random.binomial(1, 1 / (1 + np.exp(-Y)), size=n)

    full_data = pd.DataFrame({"Y": Y, "X": X, "Z": Z})

    X[R == 0] = np.nan

    corrupt_data = pd.DataFrame({"Y": Y, "X": X, "Z": Z})

    identify_dataset = Dataset()
    identify_dataset.make(corrupt_data, y="Y")
    identify_dataset.full_data = pd.DataFrame(full_data)
    return identify_dataset


def single_mar(
    n: int,
    ci: str,
) -> Dataset:
    """Generates a simple linear dataset with controllable MAR missingness.

    Args:
        n: Number of observations
        ci: Whether the data is conditionally independent (True) or not (False)

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

    if ci:
        R_latent = X1
    else:
        R_latent = Y

    R = 1 * R_latent < np.quantile(R_latent, 0.5)

    X2[R == 0] = np.nan

    corrupt_data = pd.DataFrame(
        {"Y": Y, "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5}
    )

    v4_dataset = Dataset()
    v4_dataset.make(corrupt_data, y="Y")
    v4_dataset.full_data = pd.DataFrame(full_data)
    return v4_dataset


def single_mnar(
    n: int,
    ci: str,
) -> Dataset:
    """Generates a simple linear dataset with controllable MNAR missingness.

    Args:
        n: Number of observations
        ci: Whether the data is conditionally independent (True) or not (False)

    Returns:
        A Dataset object with the full data, missing data, and

    Raises:
        ValueError: An error generating or applying missing values to the chosen column
    """
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)
    X4 = np.random.normal(0, 1, n)
    Z = np.random.normal(0, 1, n)
    Y = 5 * (X1 + X2 + X3 + X4) + np.random.normal(0, 1, n)

    full_data = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3, "X4": X4})

    if ci:
        R_latent = Z
    else:
        R_latent = Y + Z

    R = 1 * R_latent < np.quantile(R_latent, 0.5)

    X2[R == 0] = np.nan

    corrupt_data = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3, "X4": X4})

    v4_dataset = Dataset()
    v4_dataset.make(corrupt_data, y="Y")
    v4_dataset.full_data = pd.DataFrame(full_data)
    return v4_dataset


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

    MAR1_dataset = Dataset()
    MAR1_dataset.make(
        pd.DataFrame(corrupt_data, columns=["Y", "X1", "X2", "X3", "X4"]), y="Y"
    )

    MAR1_dataset.full_data = pd.DataFrame(data, columns=["Y", "X1", "X2", "X3", "X4"])

    return MAR1_dataset


def MAR1a(
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
    R_latent = data[:, 1] if ci else data[:, 0]
    M[:, 2] = ~np.all([R_latent < np.quantile(R_latent, 0.2), U3 < 0.9], axis=0)

    corrupt_data = data.copy()
    corrupt_data[~M] = np.nan

    MAR1_dataset = Dataset()
    MAR1_dataset.make(
        pd.DataFrame(corrupt_data, columns=["Y", "X1", "X2", "X3", "X4"]), y="Y"
    )

    MAR1_dataset.full_data = pd.DataFrame(data, columns=["Y", "X1", "X2", "X3", "X4"])

    return MAR1_dataset


def MNAR1(
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

    # X2 is MNAR but ci or MAR but nci
    U3 = np.random.uniform(0, 1, n)
    if ci:
        M[:, 2] = ~np.all([data[:, 2] < -1, U3 < 0.9], axis=0)
    else:
        M[:, 2] = ~np.all([data[:, 0] < -1, U3 < 0.9], axis=0)  # missing by Y

    corrupt_data = data.copy()
    corrupt_data[~M] = np.nan

    MNAR1_dataset = Dataset()
    MNAR1_dataset.make(
        pd.DataFrame(corrupt_data, columns=["Y", "X1", "X2", "X3", "X4"]), y="Y"
    )

    MNAR1_dataset.full_data = pd.DataFrame(data, columns=["Y", "X1", "X2", "X3", "X4"])

    return MNAR1_dataset


def MNAR1a(
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

    # Y is MCAR:
    M[:, 0] = U1 < 0.85

    # X3 is always observed:
    M[:, 3] = True

    # X1 is MAR:
    U2 = np.random.uniform(0, 1, n)
    M[:, 1] = ~np.all([data[:, 3] < -1, U2 < 0.9], axis=0)

    # X2 is MNAR but CIMDA/CDMDA:
    U3 = np.random.uniform(0, 1, n)
    R_latent = data[:, 4] if ci else data[:, 4] + data[:, 0]
    M[:, 2] = ~np.all([R_latent < np.quantile(R_latent, 0.2), U3 < 0.9], axis=0)

    data = data[:, :4]  # remove X4 to make it MNAR
    M = M[:, :4]  # remove X4 from mask

    corrupt_data = data.copy()
    corrupt_data[~M] = np.nan

    MNAR1_dataset = Dataset()
    MNAR1_dataset.make(
        pd.DataFrame(corrupt_data, columns=["Y", "X1", "X2", "X3"]), y="Y"
    )

    MNAR1_dataset.full_data = pd.DataFrame(data, columns=["Y", "X1", "X2", "X3"])

    return MNAR1_dataset


def MNAR1b(
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

    # Y is MCAR:
    M[:, 0] = U1 < 0.85

    # X3 is always observed:
    M[:, 3] = True

    # X1 is MAR:
    U2 = np.random.uniform(0, 1, n)
    M[:, 1] = ~np.all([data[:, 3] < -1, U2 < 0.9], axis=0)

    # X2 is MNAR but CIMDA/CDMDA:
    Z = np.random.normal(0, 1, n)
    R_latent = Z if ci else Z + data[:, 0]
    M[:, 2] = ~np.all([R_latent < np.quantile(R_latent, 0.2)], axis=0)

    corrupt_data = data.copy()
    corrupt_data[~M] = np.nan

    MNAR1_dataset = Dataset()
    MNAR1_dataset.make(
        pd.DataFrame(corrupt_data, columns=["Y", "X1", "X2", "X3", "X4"]), y="Y"
    )

    MNAR1_dataset.full_data = pd.DataFrame(data, columns=["Y", "X1", "X2", "X3", "X4"])

    return MNAR1_dataset


def adult(n=1000, ci=True, mcar_prop=0.5, k=None) -> Dataset:

    path = files("citest.data_examples").joinpath("us-census-income.csv")
    adult = pd.read_csv(path)

    if k is not None:
        # Ensure these columns are always included
        base_vars = ["income", "education", "age"]
        other_vars = [col for col in adult.columns if col not in base_vars]

        # Calculate how many additional columns to select
        k_remaining = max(0, k - len(base_vars))

        # Randomly select additional columns
        if k_remaining > 0 and len(other_vars) > 0:
            selected_cols = np.random.choice(
                a=other_vars, size=min(k_remaining, len(other_vars)), replace=False
            )
            # Combine base columns with randomly selected ones
            selected_cols = base_vars + selected_cols.tolist()
        else:
            selected_cols = base_vars

        # Filter adult dataframe to only include selected columns
        adult = adult[selected_cols]

    idxs = np.random.choice(adult.shape[0], n)
    adult_compl = adult.iloc[idxs, :]

    adult_compl.loc[:, "income"] = adult_compl["income"].map({"<=50K": 0, ">50K": 1})

    adult_compl = pd.concat(
        [adult_compl["income"], adult_compl.drop("income", axis=1)], axis=1
    ).reset_index(drop=True)

    adult_compl["income"] = adult_compl["income"].astype("int64")

    a_dataset = Dataset()
    a_dataset.make(adult_compl, y="income")
    adult_wide = a_dataset.miss_data.copy()

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
    a_dataset.full_data = adult_wide
    a_dataset.miss_data = adult_miss
    a_dataset.mask = ~adult_miss.isnull().to_numpy()

    assert a_dataset.full_data.shape == a_dataset.miss_data.shape
    assert (a_dataset.full_data.columns == a_dataset.miss_data.columns).all()

    return a_dataset


def adult_mnar(n=1000, ci=True, mcar_prop=0.5) -> Dataset:

    path = files("citest.data_examples").joinpath("us-census-income.csv")
    adult = pd.read_csv(path)

    idxs = np.random.choice(adult.shape[0], n)
    adult_compl = adult.iloc[idxs, :]

    adult_compl.loc[:, "income"] = adult_compl["income"].map({"<=50K": 0, ">50K": 1})

    adult_compl = pd.concat(
        [adult_compl["income"], adult_compl.drop("income", axis=1)], axis=1
    ).reset_index(drop=True)

    adult_compl["income"] = adult_compl["income"].astype("int64")

    adult_sex = adult_compl["sex"]
    adult_compl.drop("sex", axis=1, inplace=True)

    a_dataset = Dataset()
    a_dataset.make(adult_compl, y="income")
    adult_wide = a_dataset.miss_data.copy()

    ed_cols = list(filter(compile("^education_").match, adult_wide.columns.tolist()))

    # Missing pattern
    adult_miss = adult_wide.copy()

    if not ci:  # MNAR and NCI
        for i in range(adult_miss.shape[0]):
            if (
                adult_miss["income"].iloc[i] == 1
                and adult_sex[i] == "Male"
                and np.random.rand() < 0.9
            ):
                adult_miss.loc[i, ed_cols] = pd.NA

    else:  # MNAR and CI
        for i in range(adult_miss.shape[0]):
            if adult_sex[i] == "Male" == 1 and np.random.rand() < 0.273:
                adult_miss.loc[i, ed_cols] = pd.NA

    for c in np.random.choice(
        a=adult_miss.shape[1] - 1,
        size=int(adult_miss.shape[1] * mcar_prop),
    ):
        adult_miss.iloc[
            np.random.choice(adult_miss.shape[0], int(adult_miss.shape[0] * 0.5)), c + 1
        ] = np.nan

    # Make dataset object
    a_dataset.full_data = adult_wide
    a_dataset.miss_data = adult_miss
    a_dataset.mask = ~adult_miss.isnull().to_numpy()

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
    m_dataset.make(mushrooms_compl, y="y")
    m_dataset.full_data = m_dataset.miss_data.copy()

    odor_cols = list(
        filter(compile("^X5_").match, m_dataset.full_data.columns.tolist())
    )

    # Missing pattern
    mushrooms_miss = m_dataset.miss_data.copy().reset_index(drop=True)

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
    m_dataset.miss_data = mushrooms_miss
    m_dataset.mask = ~m_dataset.miss_data.isnull().to_numpy()

    assert m_dataset.full_data.shape == m_dataset.miss_data.shape
    assert (m_dataset.full_data.columns == m_dataset.miss_data.columns).all()

    return m_dataset
