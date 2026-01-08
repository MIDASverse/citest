import numpy as np
import pandas as pd

from typing import Optional, Dict, List
from pydantic import BaseModel, ConfigDict, PrivateAttr

from importlib.resources import files
from re import compile


def _pick_gate_col(adult_wide, ed_cols):
    banned = set(ed_cols) | {"income", "age"}
    candidates = [c for c in adult_wide.columns if c not in banned]
    return candidates[0] if len(candidates) > 0 else None


def _to_binary_gate(series):
    """Convert a pandas Series to a boolean gate."""
    s = series

    # if boolean/0-1-ish dummy, threshold at >0
    uniq = np.unique(s.dropna().to_numpy())
    if len(uniq) <= 3 and set(uniq).issubset({0, 1, False, True}):
        return s.astype(float) > 0.5

    # otherwise threshold at median
    med = np.nanmedian(s.astype(float))
    return s.astype(float) > med


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
    weights: Optional[np.ndarray] = None
    y_name: Optional[str] = None

    # private vars:
    _expl_vars: List[int] = PrivateAttr(default_factory=list)
    _raw_groups: Dict[str, List[str]] = PrivateAttr(default_factory=dict)
    _raw_groups_idx: Dict[str, List[int]] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

        data_wide = pd.get_dummies(data, dummy_na=True, drop_first=drop_first).astype(
            float
        )

        self._expl_vars = []
        self._raw_groups = {}
        self._raw_groups_idx = {}

        for col in data.columns:
            nan_col = col + "_nan"
            if nan_col in data_wide.columns:
                exp_cols = data_wide.columns.str.startswith(col + "_")

                group_cols = data_wide.columns[
                    exp_cols & ~data_wide.columns.str.endswith("_nan")
                ].tolist()

                # map raw -> OHE cols
                self._raw_groups[col] = group_cols

                if col in self.expl_vars:
                    self._expl_vars.extend(group_cols)

                data_wide.loc[data[col].isnull(), exp_cols] = np.nan

                data_wide.drop(nan_col, axis=1, inplace=True)
            else:
                self._raw_groups[col] = [col]
                if col in self.expl_vars:
                    self._expl_vars.append(col)

        return data_wide

    def _get_wgts(self):
        miss = self.miss_data.isnull().mean(axis=0).to_numpy()
        raw_wgts = miss * (1 - miss)
        self.weights = raw_wgts / raw_wgts.sum()

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

        if y not in data.columns:
            raise ValueError(
                "Outcome variable not found in data. Please provide a valid outcome variable name."
            )
        self.y_name = y
        # make sure y in first position
        data = pd.concat([data[y], data.drop(y, axis=1)], axis=1)

        if expl_vars is not None:
            self.expl_vars = expl_vars
        else:
            self.expl_vars = data.columns.tolist()[1:]

        if _onehot:
            data_wide = self._dummy(data)
        else:
            data_wide = data.copy()
            self._expl_vars = self.expl_vars.copy()
            self._raw_groups = {c: [c] for c in data.columns}

        self._raw_groups_idx = {}

        self.miss_data = data_wide
        self.mask = ~data_wide.isnull().to_numpy()
        self.n = data_wide.shape[0]
        self.full_data = None

        self._get_wgts()

        self._expl_vars = data_wide.columns.get_indexer(self._expl_vars).tolist()
        self._raw_groups_idx = {
            k: data_wide.columns.get_indexer(v).tolist()
            for k, v in self._raw_groups.items()
        }

    def get_predictor_cols_idx(self) -> List[int]:
        """Indices of predictors used in the test: [Y] + expanded expl vars."""
        return [0] + self._expl_vars

    def get_target_mask(self, level: str = "column") -> np.ndarray:
        """
        level="column": targets are per-wide-column (OHE columns)
        level="variable": targets are per-raw-variable (Outcome + expl_vars)
        """
        if level == "column":
            cols_idx = self.get_predictor_cols_idx()
            return self.mask[:, cols_idx].astype(float)

        if level == "variable":
            # raw variables to score: outcome + raw expl_vars
            raw_vars = [self.y_name] + list(self.expl_vars)
            groups = [self._raw_groups_idx[v] for v in raw_vars]

            # variable is observed iff all its wide columns are observed
            return np.stack(
                [self.mask[:, g].all(axis=1) for g in groups], axis=1
            ).astype(float)

        raise ValueError("level must be 'column' or 'variable'")

    def get_target_weights(self, level: str = "column") -> np.ndarray:
        """
        Weights aligned with get_target_mask(level=...).
        For variable-level: compute weights from variable-level missingness.
        """
        if level == "column":
            cols_idx = self.get_predictor_cols_idx()
            w = self.weights
            if w is None or not np.isfinite(w).all():
                return np.full(len(cols_idx), 1.0 / len(cols_idx))
            w = w[cols_idx]
            return w / w.sum()

        if level == "variable":
            mask_var = self.get_target_mask(level="variable")
            miss = 1.0 - mask_var.mean(axis=0)
            raw_w = miss * (1.0 - miss)
            if not np.isfinite(raw_w).all() or raw_w.sum() == 0:
                return np.full(mask_var.shape[1], 1.0 / mask_var.shape[1])
            return raw_w / raw_w.sum()

        raise ValueError("level must be 'column' or 'variable'")


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

    R = (R_latent < np.quantile(R_latent, 0.5)).astype(int)

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

    R = (R_latent < np.quantile(R_latent, 0.5)).astype(int)

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
    ci: bool,
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
    ci: bool,
    missing_mech: str = "linear",
) -> Dataset:
    """Generates a simple linear dataset with controllable MAR missingness.

    Args:
        n: Number of observations
        ci: Whether the data is conditionally independent (True) or not (False)
        missing_mech: Whether the missing mechanism is "linear"" in {X,Y} or uses non-linear + XOR transformations

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
    Y = np.random.normal(0, 1, n)

    full_data = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5})

    if missing_mech.lower() == "linear":
        R_latent = X1 if ci else Y
    elif missing_mech.lower() == "xor":
        X_xor = (X1 > 0).astype(int) ^ (X3 > 0).astype(int)
        X_xor = 2.0 * X_xor - 1.0
        eps = 0.1 * np.random.normal(size=n)  # break ties, make latent continuous

        if ci:
            # CI: depends only on X (still MAR since X observed)
            R_latent = X_xor + 0.5 * np.sin(X4) + eps
        else:
            # NCI: add Y-dependent interaction (still MAR since Y observed)
            Y_xor = (Y > np.median(Y)).astype(int) ^ (X5 > 0).astype(int)
            Y_xor = 2.0 * Y_xor - 1.0
            R_latent = X_xor + 0.5 * np.sin(X4) + 1.0 * Y_xor + eps
    else:
        raise ValueError("missing_mech must be one of 'linear' or 'XOR'")

    R = (R_latent < np.quantile(R_latent, 0.5)).astype(int)

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
    ci: bool,
    missing_mech: str = "linear",
) -> Dataset:
    """Generates a simple linear dataset with controllable MNAR missingness.

    Args:
        n: Number of observations
        ci: Whether the data is conditionally independent (True) or not (False)
        missing_mech: Whether the missing mechanism is "linear"" in {X,Y} or uses non-linear + XOR transformations

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
    Y = np.random.normal(0, 1, n)

    full_data = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3, "X4": X4})

    if missing_mech.lower() == "linear":
        R_latent = Z if ci else Y + Z

    elif missing_mech.lower() == "xor":
        eps = 0.1 * np.random.normal(size=n)

        if ci:
            R_latent = np.sin(2 * Z) + eps
        else:
            # NCI: add Y-dependent XOR interaction (still includes Z)
            Y_xor = (Y > np.median(Y)).astype(int) ^ (X1 > 0).astype(int)
            Y_xor = 2.0 * Y_xor - 1.0
            R_latent = np.sin(2 * Z) + 1.5 * Y_xor + eps
    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    R = (R_latent < np.quantile(R_latent, 0.5)).astype(int)

    X2[R == 0] = np.nan

    corrupt_data = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3, "X4": X4})

    v4_dataset = Dataset()
    v4_dataset.make(corrupt_data, y="Y")
    v4_dataset.full_data = pd.DataFrame(full_data)
    return v4_dataset


def MAR1(
    n: int,
    ci: bool = True,
    missing_mech: str = "linear",
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
    M[:, 0] = True  # fully observed to preserve MAR condition
    M[:, 4] = U1 < 0.85

    # X3 is always observed:
    M[:, 3] = True

    # X1 is MAR:
    U2 = np.random.uniform(0, 1, n)
    M[:, 1] = ~np.all([data[:, 3] < -1, U2 < 0.9], axis=0)

    # X2 is MAR but variable:
    U3 = np.random.uniform(0, 1, n)
    mech = missing_mech.lower()
    if mech == "linear":
        R_latent = data[:, 3] if ci else data[:, 0]
    elif mech == "xor":
        y = data[:, 0]
        x1 = data[:, 1]
        x3 = data[:, 3]

        eps = 0.05 * np.random.normal(size=n)

        # baseline nonlinear in X-only (CI case) so still MAR
        # includes XOR-like gate that plain logistic on raw vars won't model well
        gate_X = (x1 > 0).astype(int) ^ (x3 > 0).astype(int)
        gate_X = 2.0 * gate_X - 1.0

        base = gate_X + 0.6 * np.sin(1.5 * x3) + 0.2 * np.cos(1.2 * x1) + eps

        if ci:
            R_latent = base
        else:
            # add Y-dependent interaction: XOR between sign(Y) and sign(X3)
            gate_Y = (y > 0).astype(int) ^ (x3 > 0).astype(int)
            gate_Y = 2.0 * gate_Y - 1.0
            R_latent = base + 1.0 * gate_Y

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

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
    missing_mech: str = "linear",
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

    # Y is always observed:
    M[:, 0] = True
    # X4 is MCAR
    M[:, 4] = U1 < 0.85

    # X3 is always observed:
    M[:, 3] = True

    # X1 is MAR:
    U2 = np.random.uniform(0, 1, n)
    M[:, 1] = ~np.all([data[:, 3] < -1, U2 < 0.9], axis=0)

    # X2 is MNAR but CIMDA/CDMDA:
    U3 = np.random.uniform(0, 1, n)
    Z = np.random.normal(0, 1, n)

    mech = missing_mech.lower()
    if mech == "linear":
        R_latent = Z if ci else Z + 2 * data[:, 0]
    elif mech == "xor":
        y = data[:, 0]
        x3 = data[:, 3]
        eps = 0.05 * np.random.normal(size=n)

        base = np.sin(2.0 * Z) + 0.2 * np.cos(1.5 * Z) + eps  # nonlinear Z-only

        if ci:
            R_latent = base
        else:
            gate_Y = (y > 0).astype(int) ^ (x3 > 0).astype(int)
            gate_Y = 2.0 * gate_Y - 1.0
            R_latent = base + 1.2 * gate_Y

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    M[:, 2] = ~np.all([R_latent < np.quantile(R_latent, 0.2), U3 < 0.9], axis=0)

    corrupt_data = data.copy()
    corrupt_data[~M] = np.nan

    MNAR1_dataset = Dataset()
    MNAR1_dataset.make(
        pd.DataFrame(corrupt_data, columns=["Y", "X1", "X2", "X3", "X4"]), y="Y"
    )

    MNAR1_dataset.full_data = pd.DataFrame(data, columns=["Y", "X1", "X2", "X3", "X4"])

    return MNAR1_dataset


def adult(
    n=1000, ci=True, mcar_prop=0.5, k=None, missing_mech: str = "linear"
) -> Dataset:

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
    with np.errstate(divide="ignore", invalid="ignore"):
        a_dataset.make(adult_compl, y="income")
    adult_wide = a_dataset.miss_data.copy()

    ed_cols = list(filter(compile("^education_").match, adult_wide.columns.tolist()))

    # Missing pattern
    adult_miss = adult_wide.copy()

    mech = missing_mech.lower()
    rng_u = np.random.rand(adult_miss.shape[0])

    y = adult_miss["income"].astype(int)
    age = adult_miss["age"].astype(float)

    if mech == "linear":
        if ci:
            trigger = age <= 30
        else:
            trigger = y == 1

    elif mech == "xor":
        # Primary gate from age (keep the same "age <= 30" flavour)
        g1 = age <= 30

        gate_col = _pick_gate_col(adult_miss, ed_cols)
        if gate_col is None:
            # fallback: nonlinear 1D gate from age decile parity (hard for plain logistic)
            # e.g., alternating decades: 0–9,10–19,... -> parity
            decade = (np.floor(age / 10.0)).astype(int)
            g2 = decade % 2 == 0
        else:
            g2 = _to_binary_gate(adult_miss[gate_col])

        base = g1 ^ g2  # XOR: not linearly separable in (g1,g2)

        if ci:
            trigger = base
        else:
            # add Y in a nonlinear way: parity flip by income
            trigger = base ^ (y == 1)

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    adult_miss.loc[miss_rows, ed_cols] = pd.NA

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

    a_dataset._get_wgts()

    assert a_dataset.full_data.shape == a_dataset.miss_data.shape
    assert (a_dataset.full_data.columns == a_dataset.miss_data.columns).all()

    return a_dataset


def adult_mnar(n=1000, ci=True, mcar_prop=0.5, missing_mech: str = "linear") -> Dataset:

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
    with np.errstate(divide="ignore", invalid="ignore"):
        a_dataset.make(adult_compl, y="income")
    adult_wide = a_dataset.miss_data.copy()

    ed_cols = list(filter(compile("^education_").match, adult_wide.columns.tolist()))

    # Missing pattern
    adult_miss = adult_wide.copy()

    mech = missing_mech.lower()
    rng_u = np.random.rand(adult_miss.shape[0])

    y = adult_miss["income"].astype(int)
    age = adult_miss["age"].astype(float) if "age" in adult_miss.columns else None
    male = adult_sex.to_numpy() == "Male"

    if mech == "linear":
        if not ci:
            miss_rows = (y.to_numpy() == 1) & male & (rng_u < 0.9)
        else:
            miss_rows = male & (rng_u < 0.273)

    elif mech == "xor":
        # Build an observed gate g2 from non-education features (sex is latent)
        gate_col = _pick_gate_col(adult_miss, ed_cols)
        if gate_col is None and age is not None:
            decade = (np.floor(age / 10.0)).astype(int)
            g2 = (decade % 2 == 0).to_numpy()
        elif gate_col is not None:
            g2 = _to_binary_gate(adult_miss[gate_col]).to_numpy()
        else:
            g2 = np.random.rand(adult_miss.shape[0]) < 0.5

        base = male ^ g2  # XOR between latent sex and observed covariate gate

        if ci:
            # CI: depends on latent sex + observed gate, not on income
            miss_rows = base & (rng_u < 0.9)
        else:
            # NCI: flip parity by income => introduces Y effect in a nonlinear way
            miss_rows = (base ^ (y.to_numpy() == 1)) & (rng_u < 0.9)

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    adult_miss.loc[miss_rows, ed_cols] = pd.NA

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

    a_dataset._get_wgts()

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

    m_dataset._get_wgts()

    return m_dataset
