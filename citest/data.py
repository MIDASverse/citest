import numpy as np
import pandas as pd
from pathlib import Path

from typing import Optional, Dict, List
from pydantic import BaseModel, ConfigDict, PrivateAttr

from importlib.resources import files
from re import compile, escape


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


def _get_cache_dir() -> Path:
    """Shared cache directory for downloaded datasets."""
    cache_dir = Path.home() / ".cache" / "citest_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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
        w_sum = raw_wgts.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            self.weights = np.full_like(raw_wgts, 1.0 / raw_wgts.size, dtype=float)
        else:
            self.weights = raw_wgts / w_sum

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
            R_latent = base + 5.0 * gate_Y

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
            R_latent = base + 3.0 * gate_Y

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
    n=1000,
    ci=True,
    mcar_prop=0.5,
    k=None,
    missing_mech: str = "linear",
    beta_y: float = 6.0,
) -> Dataset:

    path = files("citest.data_examples").joinpath("us-census-income.csv")
    adult = pd.read_csv(path)

    # for knockoff testing purposes, limit to k columns if specified
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
        age_np = age.to_numpy()
        y_np = y.to_numpy()

        g1 = age_np <= 30

        gate_col = _pick_gate_col(adult_miss, ed_cols)
        if gate_col is None:
            decade = np.floor(age_np / 10.0).astype(int)
            g2 = decade % 2 == 0
        else:
            g2 = _to_binary_gate(adult_miss[gate_col]).to_numpy()

        eps = 0.05 * np.random.normal(size=adult_miss.shape[0])
        gate_X = (g1 ^ g2).astype(float)
        gate_X = 2.0 * gate_X - 1.0

        base = gate_X + 0.3 * np.sin(age_np / 10.0) + eps

        if ci:
            R_latent = base
        else:
            gate_Y = ((y_np == 1) ^ g2).astype(float)
            gate_Y = 2.0 * gate_Y - 1.0
            R_latent = base + beta_y * gate_Y

        trigger = R_latent < np.quantile(R_latent, 0.2)

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    adult_miss.loc[miss_rows, ed_cols] = pd.NA

    raw_cols = [
        c for c in adult_compl.columns if c not in ["income", "education", "age"]
    ]
    for c in np.random.choice(
        a=raw_cols,
        size=int(len(raw_cols) * mcar_prop),
        replace=False,
    ):
        var_pat = compile(rf"^{escape(c)}_")
        mcar_cols = [col for col in adult_wide.columns if var_pat.match(col)]
        if len(mcar_cols) == 0 and c in adult_miss.columns:
            mcar_cols = [c]
        mcar_rows = np.random.choice(
            adult_miss.shape[0], int(adult_miss.shape[0] * 0.5), replace=False
        )
        adult_miss.loc[mcar_rows, mcar_cols] = np.nan

    # for c in np.random.choice(
    #     a=adult_miss.shape[1] - 1,
    #     size=int(adult_miss.shape[1] * mcar_prop),
    # ):
    #     adult_miss.iloc[
    #         np.random.choice(adult_miss.shape[0], int(adult_miss.shape[0] * 0.5)), c + 1
    #     ] = np.nan

    # Make dataset object
    a_dataset.full_data = adult_wide
    a_dataset.miss_data = adult_miss
    a_dataset.mask = ~adult_miss.isnull().to_numpy()

    a_dataset._get_wgts()

    assert a_dataset.full_data.shape == a_dataset.miss_data.shape
    assert (a_dataset.full_data.columns == a_dataset.miss_data.columns).all()

    return a_dataset


def adult_mnar(
    n=1000, ci=True, mcar_prop=0.5, missing_mech: str = "linear", beta_y: float = 6.0
) -> Dataset:

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
        y_np = y.to_numpy()
        age_np = age.to_numpy() if age is not None else None

        gate_col = _pick_gate_col(adult_miss, ed_cols)
        if gate_col is None and age_np is not None:
            decade = np.floor(age_np / 10.0).astype(int)
            g2 = decade % 2 == 0
        elif gate_col is not None:
            g2 = _to_binary_gate(adult_miss[gate_col]).to_numpy()
        else:
            g2 = np.random.rand(adult_miss.shape[0]) < 0.5

        eps = 0.05 * np.random.normal(size=adult_miss.shape[0])
        gate_X = (male ^ g2).astype(float)
        gate_X = 2.0 * gate_X - 1.0
        base = gate_X + eps

        if ci:
            R_latent = base
        else:
            gate_Y = ((y_np == 1) ^ g2).astype(float)
            gate_Y = 2.0 * gate_Y - 1.0
            R_latent = base + beta_y * gate_Y

        trigger = R_latent < np.quantile(R_latent, 0.2)
        miss_rows = trigger & (rng_u < 0.9)

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    adult_miss.loc[miss_rows, ed_cols] = pd.NA

    raw_cols = [
        c for c in adult_compl.columns if c not in ["income", "education", "age"]
    ]
    for c in np.random.choice(
        a=raw_cols,
        size=int(len(raw_cols) * mcar_prop),
        replace=False,
    ):
        var_pat = compile(rf"^{escape(c)}_")
        mcar_cols = [col for col in adult_wide.columns if var_pat.match(col)]
        if len(mcar_cols) == 0 and c in adult_miss.columns:
            mcar_cols = [c]
        mcar_rows = np.random.choice(
            adult_miss.shape[0], int(adult_miss.shape[0] * 0.5), replace=False
        )
        adult_miss.loc[mcar_rows, mcar_cols] = np.nan

    # for c in np.random.choice(
    #     a=adult_miss.shape[1] - 1,
    #     size=int(adult_miss.shape[1] * mcar_prop),
    # ):
    #     adult_miss.iloc[
    #         np.random.choice(adult_miss.shape[0], int(adult_miss.shape[0] * 0.5)), c + 1
    #     ] = np.nan

    # Make dataset object
    a_dataset.full_data = adult_wide
    a_dataset.miss_data = adult_miss
    a_dataset.mask = ~adult_miss.isnull().to_numpy()

    a_dataset._get_wgts()

    assert a_dataset.full_data.shape == a_dataset.miss_data.shape
    assert (a_dataset.full_data.columns == a_dataset.miss_data.columns).all()

    return a_dataset


def mushrooms(n=1000, ci=True, mcar_prop=0.5, missing_mech: str = "linear") -> Dataset:

    path = files("citest.data_examples").joinpath("agaricus-lepiota.data")
    mushrooms = pd.read_csv(path, delimiter=",", header=None)
    mushrooms.columns = ["y"] + [f"X{i}" for i in range(1, mushrooms.shape[1])]
    idxs = np.random.choice(mushrooms.shape[0], n)

    mushrooms_compl = mushrooms.iloc[idxs, :].copy().reset_index(drop=True)

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
    mushrooms_miss = m_dataset.miss_data.copy()
    rng_u = np.random.rand(mushrooms_miss.shape[0])

    mech = missing_mech.lower()
    if mech == "linear":
        if ci:
            trigger = (mushrooms_compl["X4"] == "t").to_numpy()
        else:
            trigger = mushrooms_miss["y"].to_numpy() == 1

    elif mech == "xor":
        # gate 1: bruises flag (observed X4)
        g1 = (mushrooms_compl["X4"] == "t").to_numpy()

        # gate 2: choose any observed non-odor covariate after OHE
        gate_candidates = [
            c
            for c in m_dataset.full_data.columns
            if c not in (["y"] + odor_cols) and not c.startswith("X4_")
        ]
        if len(gate_candidates) == 0:
            g2 = np.random.rand(mushrooms_miss.shape[0]) < 0.5
        else:
            g2 = _to_binary_gate(m_dataset.full_data[gate_candidates[0]]).to_numpy()

        base = g1 ^ g2  # XOR to create non-linear gate

        if ci:
            trigger = base
        else:
            # add outcome parity flip to inject Y dependence nonlinearly
            trigger = base ^ (mushrooms_miss["y"].to_numpy() == 1)

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    mushrooms_miss.loc[miss_rows, odor_cols] = pd.NA

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


def breast_cancer(
    n=500, ci=True, mcar_prop=0.5, missing_mech: str = "linear"
) -> Dataset:
    """Wisconsin Diagnostic Breast Cancer dataset with MAR/MNAR-style masks."""

    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "y"})
    idxs = np.random.choice(data.shape[0], n)
    compl = data.iloc[idxs, :].copy().reset_index(drop=True)
    compl["y"] = compl["y"].astype(int)

    bc_dataset = Dataset()
    bc_dataset.make(compl, y="y")
    bc_dataset.full_data = bc_dataset.miss_data.copy()

    mask_cols = [c for c in bc_dataset.full_data.columns if c.startswith("worst")]
    if len(mask_cols) == 0:
        mask_cols = bc_dataset.full_data.columns[1:6].tolist()

    bc_miss = bc_dataset.miss_data.copy()
    rng_u = np.random.rand(bc_miss.shape[0])

    tex_col = (
        "mean texture"
        if "mean texture" in compl.columns
        else bc_dataset.full_data.columns[1]
    )
    gate_col = (
        "perimeter error"
        if "perimeter error" in compl.columns
        else bc_dataset.full_data.columns[min(2, bc_dataset.full_data.shape[1] - 1)]
    )

    mech = missing_mech.lower()
    if mech == "linear":
        if ci:
            trigger = (compl[tex_col] > compl[tex_col].median()).to_numpy()
        else:
            trigger = compl["y"].to_numpy() == 1

    elif mech == "xor":
        g1 = (compl[tex_col] > compl[tex_col].median()).to_numpy()
        g2 = _to_binary_gate(bc_dataset.full_data[gate_col]).to_numpy()
        base = g1 ^ g2

        if ci:
            trigger = base
        else:
            trigger = base ^ (compl["y"].to_numpy() == 1)

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    bc_miss.loc[miss_rows, mask_cols] = pd.NA

    for c in np.random.choice(
        a=bc_miss.shape[1] - 1,
        size=int(bc_miss.shape[1] * mcar_prop),
    ):
        bc_miss.iloc[
            np.random.choice(bc_miss.shape[0], int(bc_miss.shape[0] * 0.5)), c + 1
        ] = np.nan

    bc_dataset.miss_data = bc_miss
    bc_dataset.mask = ~bc_dataset.miss_data.isnull().to_numpy()

    assert bc_dataset.full_data.shape == bc_dataset.miss_data.shape
    assert (bc_dataset.full_data.columns == bc_dataset.miss_data.columns).all()

    bc_dataset._get_wgts()

    return bc_dataset


def wine(n=500, ci=True, mcar_prop=0.5, missing_mech: str = "linear") -> Dataset:
    """UCI Wine dataset with controllable MAR/MNAR masking."""

    from sklearn.datasets import load_wine

    data = load_wine(as_frame=True).frame.rename(columns={"target": "y"})
    idxs = np.random.choice(data.shape[0], n)
    compl = data.iloc[idxs, :].copy().reset_index(drop=True)
    compl["y"] = compl["y"].astype(int)

    w_dataset = Dataset()
    w_dataset.make(compl, y="y")
    w_dataset.full_data = w_dataset.miss_data.copy()

    mask_cols = [
        c for c in w_dataset.full_data.columns if "phenols" in c or "color" in c
    ]
    if len(mask_cols) == 0:
        mask_cols = w_dataset.full_data.columns[1:5].tolist()

    w_miss = w_dataset.miss_data.copy()
    rng_u = np.random.rand(w_miss.shape[0])

    hue_col = "hue" if "hue" in compl.columns else w_dataset.full_data.columns[1]
    pro_col = (
        "proline"
        if "proline" in compl.columns
        else w_dataset.full_data.columns[min(2, w_dataset.full_data.shape[1] - 1)]
    )

    mech = missing_mech.lower()
    if mech == "linear":
        if ci:
            trigger = (compl[hue_col] > compl[hue_col].median()).to_numpy()
        else:
            trigger = compl["y"].to_numpy() == 2

    elif mech == "xor":
        g1 = (compl[hue_col] > compl[hue_col].median()).to_numpy()
        g2 = _to_binary_gate(w_dataset.full_data[pro_col]).to_numpy()
        base = g1 ^ g2

        if ci:
            trigger = base
        else:
            trigger = base ^ (compl["y"].to_numpy() == 2)

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    w_miss.loc[miss_rows, mask_cols] = pd.NA

    for c in np.random.choice(
        a=w_miss.shape[1] - 1,
        size=int(w_miss.shape[1] * mcar_prop),
    ):
        w_miss.iloc[
            np.random.choice(w_miss.shape[0], int(w_miss.shape[0] * 0.5)), c + 1
        ] = np.nan

    w_dataset.miss_data = w_miss
    w_dataset.mask = ~w_dataset.miss_data.isnull().to_numpy()

    assert w_dataset.full_data.shape == w_dataset.miss_data.shape
    assert (w_dataset.full_data.columns == w_dataset.miss_data.columns).all()

    w_dataset._get_wgts()

    return w_dataset


def diabetes(n=442, ci=True, mcar_prop=0.5, missing_mech: str = "linear") -> Dataset:
    """Diabetes progression dataset with MAR/MNAR masks (regression target)."""

    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True).frame.rename(columns={"target": "y"})
    idxs = np.random.choice(data.shape[0], n)
    compl = data.iloc[idxs, :].copy().reset_index(drop=True)
    compl["y"] = compl["y"].astype(float)

    d_dataset = Dataset()
    d_dataset.make(compl, y="y")
    d_dataset.full_data = d_dataset.miss_data.copy()

    mask_cols = [c for c in d_dataset.full_data.columns if c in {"bmi", "bp", "s5"}]
    if len(mask_cols) == 0:
        mask_cols = d_dataset.full_data.columns[1:5].tolist()

    d_miss = d_dataset.miss_data.copy()
    rng_u = np.random.rand(d_miss.shape[0])

    bmi_col = "bmi" if "bmi" in compl.columns else d_dataset.full_data.columns[1]
    age_col = (
        "age"
        if "age" in compl.columns
        else d_dataset.full_data.columns[min(2, d_dataset.full_data.shape[1] - 1)]
    )

    mech = missing_mech.lower()
    if mech == "linear":
        if ci:
            trigger = (compl[bmi_col] > compl[bmi_col].median()).to_numpy()
        else:
            trigger = compl["y"].to_numpy() > np.median(compl["y"])

    elif mech == "xor":
        g1 = (compl[age_col] > compl[age_col].median()).to_numpy()
        g2 = _to_binary_gate(compl[bmi_col]).to_numpy()
        base = g1 ^ g2

        if ci:
            trigger = base
        else:
            trigger = base ^ (compl["y"].to_numpy() > np.median(compl["y"]))

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    d_miss.loc[miss_rows, mask_cols] = pd.NA

    for c in np.random.choice(
        a=d_miss.shape[1] - 1,
        size=int(d_miss.shape[1] * mcar_prop),
    ):
        d_miss.iloc[
            np.random.choice(d_miss.shape[0], int(d_miss.shape[0] * 0.5)), c + 1
        ] = np.nan

    d_dataset.miss_data = d_miss
    d_dataset.mask = ~d_dataset.miss_data.isnull().to_numpy()

    assert d_dataset.full_data.shape == d_dataset.miss_data.shape
    assert (d_dataset.full_data.columns == d_dataset.miss_data.columns).all()

    d_dataset._get_wgts()

    return d_dataset


def covertype(n=5000, ci=True, mcar_prop=0.3, missing_mech: str = "linear") -> Dataset:
    """Forest CoverType dataset with MAR/MNAR-style masking."""

    from sklearn.datasets import fetch_covtype

    data = fetch_covtype(as_frame=True, data_home=_get_cache_dir())
    frame = data.frame.copy()
    frame = frame.rename(columns={"Cover_Type": "y"})

    n = min(n, frame.shape[0])
    idxs = np.random.choice(frame.shape[0], n, replace=False)
    compl = frame.iloc[idxs, :].reset_index(drop=True)
    compl["y"] = compl["y"].astype(int)

    cv_dataset = Dataset()
    cv_dataset.make(compl, y="y")
    cv_dataset.full_data = cv_dataset.miss_data.copy()

    mask_cols = [
        c for c in cv_dataset.full_data.columns if "Soil" in c or "Wilderness" in c
    ]
    if len(mask_cols) == 0:
        mask_cols = cv_dataset.full_data.columns[1:6].tolist()

    cv_miss = cv_dataset.miss_data.copy()
    rng_u = np.random.rand(cv_miss.shape[0])

    elev_col = (
        "Elevation" if "Elevation" in compl.columns else cv_dataset.full_data.columns[1]
    )
    aspect_col = (
        "Aspect"
        if "Aspect" in compl.columns
        else cv_dataset.full_data.columns[min(2, cv_dataset.full_data.shape[1] - 1)]
    )

    mech = missing_mech.lower()
    if mech == "linear":
        if ci:
            trigger = (compl[elev_col] > compl[elev_col].median()).to_numpy()
        else:
            trigger = compl["y"].to_numpy() == 1

    elif mech == "xor":
        g1 = (compl[elev_col] > compl[elev_col].median()).to_numpy()
        g2 = _to_binary_gate(cv_dataset.full_data[aspect_col]).to_numpy()
        base = g1 ^ g2
        if ci:
            trigger = base
        else:
            trigger = base ^ (compl["y"].to_numpy() == 1)

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    cv_miss.loc[miss_rows, mask_cols] = pd.NA

    for c in np.random.choice(
        a=cv_miss.shape[1] - 1,
        size=int(cv_miss.shape[1] * mcar_prop),
    ):
        cv_miss.iloc[
            np.random.choice(cv_miss.shape[0], int(cv_miss.shape[0] * 0.5)), c + 1
        ] = np.nan

    cv_dataset.miss_data = cv_miss
    cv_dataset.mask = ~cv_dataset.miss_data.isnull().to_numpy()

    assert cv_dataset.full_data.shape == cv_dataset.miss_data.shape
    assert (cv_dataset.full_data.columns == cv_dataset.miss_data.columns).all()

    cv_dataset._get_wgts()

    return cv_dataset


def california_housing(
    n=20000, ci=True, mcar_prop=0.3, missing_mech: str = "linear"
) -> Dataset:
    """California Housing regression dataset with MAR/MNAR masks."""

    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True, data_home=_get_cache_dir())
    frame = data.frame.copy()
    frame = frame.rename(columns={"MedHouseVal": "y"})

    n = min(n, frame.shape[0])
    idxs = np.random.choice(frame.shape[0], n, replace=False)
    compl = frame.iloc[idxs, :].reset_index(drop=True)
    compl["y"] = compl["y"].astype(float)

    ca_dataset = Dataset()
    ca_dataset.make(compl, y="y")
    ca_dataset.full_data = ca_dataset.miss_data.copy()

    mask_cols = [
        c
        for c in ca_dataset.full_data.columns
        if c in {"MedInc", "AveRooms", "AveBedrms"}
    ]
    if len(mask_cols) == 0:
        mask_cols = ca_dataset.full_data.columns[1:5].tolist()

    ca_miss = ca_dataset.miss_data.copy()
    rng_u = np.random.rand(ca_miss.shape[0])

    age_col = (
        "HouseAge" if "HouseAge" in compl.columns else ca_dataset.full_data.columns[1]
    )
    lat_col = (
        "Latitude"
        if "Latitude" in compl.columns
        else ca_dataset.full_data.columns[min(2, ca_dataset.full_data.shape[1] - 1)]
    )

    mech = missing_mech.lower()
    if mech == "linear":
        if ci:
            trigger = (compl[age_col] > compl[age_col].median()).to_numpy()
        else:
            trigger = compl["y"].to_numpy() > np.median(compl["y"])

    elif mech == "xor":
        g1 = (compl[age_col] > compl[age_col].median()).to_numpy()
        g2 = _to_binary_gate(ca_dataset.full_data[lat_col]).to_numpy()
        base = g1 ^ g2
        if ci:
            trigger = base
        else:
            trigger = base ^ (compl["y"].to_numpy() > np.median(compl["y"]))

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    ca_miss.loc[miss_rows, mask_cols] = pd.NA

    for c in np.random.choice(
        a=ca_miss.shape[1] - 1,
        size=int(ca_miss.shape[1] * mcar_prop),
    ):
        ca_miss.iloc[
            np.random.choice(ca_miss.shape[0], int(ca_miss.shape[0] * 0.5)), c + 1
        ] = np.nan

    ca_dataset.miss_data = ca_miss
    ca_dataset.mask = ~ca_dataset.miss_data.isnull().to_numpy()

    assert ca_dataset.full_data.shape == ca_dataset.miss_data.shape
    assert (ca_dataset.full_data.columns == ca_dataset.miss_data.columns).all()

    ca_dataset._get_wgts()

    return ca_dataset


def german_credit(
    n=1000, ci=True, mcar_prop=0.3, missing_mech: str = "linear"
) -> Dataset:
    """German credit (OpenML id=31) with MAR/MNAR masking on categorical features."""

    from sklearn.datasets import fetch_openml

    data = fetch_openml(
        "credit-g", version=1, as_frame=True, data_home=_get_cache_dir()
    )
    frame = data.frame.copy()
    frame = frame.rename(columns={"class": "y"})
    frame["y"] = frame["y"].map({"good": 1, "bad": 0})

    n = min(n, frame.shape[0])
    idxs = np.random.choice(frame.shape[0], n, replace=False)
    compl = frame.iloc[idxs, :].reset_index(drop=True)
    compl["y"] = compl["y"].astype(int)
    compl = pd.concat([compl["y"], compl.drop(columns=["y"])], axis=1)

    gc_dataset = Dataset()
    gc_dataset.make(compl, y="y")
    gc_dataset.full_data = gc_dataset.miss_data.copy()

    mask_cols = [
        c for c in gc_dataset.full_data.columns if "checking" in c or "saving" in c
    ]
    if len(mask_cols) == 0:
        mask_cols = gc_dataset.full_data.columns[1:5].tolist()

    gc_miss = gc_dataset.miss_data.copy()
    rng_u = np.random.rand(gc_miss.shape[0])

    dur_col = (
        "duration" if "duration" in compl.columns else gc_dataset.full_data.columns[1]
    )
    amt_col = (
        "credit_amount"
        if "credit_amount" in compl.columns
        else gc_dataset.full_data.columns[min(2, gc_dataset.full_data.shape[1] - 1)]
    )

    mech = missing_mech.lower()
    if mech == "linear":
        if ci:
            trigger = (compl[dur_col] > compl[dur_col].median()).to_numpy()
        else:
            trigger = compl["y"].to_numpy() == 1

    elif mech == "xor":
        g1 = (compl[dur_col] > compl[dur_col].median()).to_numpy()
        amt_numeric_full = pd.to_numeric(
            gc_dataset.full_data[amt_col], errors="coerce"
        ).fillna(0)
        g2 = _to_binary_gate(amt_numeric_full).to_numpy()
        base = g1 ^ g2
        if ci:
            trigger = base
        else:
            trigger = base ^ (compl["y"].to_numpy() == 1)

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    gc_miss.loc[miss_rows, mask_cols] = pd.NA

    for c in np.random.choice(
        a=gc_miss.shape[1] - 1,
        size=int(gc_miss.shape[1] * mcar_prop),
    ):
        gc_miss.iloc[
            np.random.choice(gc_miss.shape[0], int(gc_miss.shape[0] * 0.5)), c + 1
        ] = np.nan

    gc_dataset.miss_data = gc_miss
    gc_dataset.mask = ~gc_dataset.miss_data.isnull().to_numpy()

    assert gc_dataset.full_data.shape == gc_dataset.miss_data.shape
    assert (gc_dataset.full_data.columns == gc_dataset.miss_data.columns).all()

    gc_dataset._get_wgts()

    return gc_dataset


def bank_marketing(
    n=10000, ci=True, mcar_prop=0.3, missing_mech: str = "linear"
) -> Dataset:
    """Bank marketing (OpenML id=1461) with MAR/MNAR masking."""

    from sklearn.datasets import fetch_openml

    data = fetch_openml(
        "bank-marketing", version=1, as_frame=True, data_home=_get_cache_dir()
    )
    frame = data.frame.copy()
    target_col = "y" if "y" in frame.columns else data.target_names[0]
    frame = frame.rename(columns={target_col: "y"})
    frame["y"] = frame["y"].map({"yes": 1, "no": 0})

    n = min(n, frame.shape[0])
    idxs = np.random.choice(frame.shape[0], n, replace=False)
    compl = frame.iloc[idxs, :].reset_index(drop=True)

    bm_dataset = Dataset()
    bm_dataset.make(compl, y="y")
    bm_dataset.full_data = bm_dataset.miss_data.copy()

    mask_cols = [
        c for c in bm_dataset.full_data.columns if "contact" in c or "poutcome" in c
    ]
    if len(mask_cols) == 0:
        mask_cols = bm_dataset.full_data.columns[1:6].tolist()

    bm_miss = bm_dataset.miss_data.copy()
    rng_u = np.random.rand(bm_miss.shape[0])

    age_col = "age" if "age" in compl.columns else bm_dataset.full_data.columns[1]
    dur_col = (
        "duration"
        if "duration" in compl.columns
        else bm_dataset.full_data.columns[min(2, bm_dataset.full_data.shape[1] - 1)]
    )

    mech = missing_mech.lower()
    if mech == "linear":
        if ci:
            trigger = (compl[age_col] > compl[age_col].median()).to_numpy()
        else:
            trigger = compl["y"].to_numpy() == 1

    elif mech == "xor":
        g1 = (compl[age_col] > compl[age_col].median()).to_numpy()
        g2 = _to_binary_gate(bm_dataset.full_data[dur_col]).to_numpy()
        base = g1 ^ g2
        if ci:
            trigger = base
        else:
            trigger = base ^ (compl["y"].to_numpy() == 1)

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    bm_miss.loc[miss_rows, mask_cols] = pd.NA

    for c in np.random.choice(
        a=bm_miss.shape[1] - 1,
        size=int(bm_miss.shape[1] * mcar_prop),
    ):
        bm_miss.iloc[
            np.random.choice(bm_miss.shape[0], int(bm_miss.shape[0] * 0.5)), c + 1
        ] = np.nan

    bm_dataset.miss_data = bm_miss
    bm_dataset.mask = ~bm_dataset.miss_data.isnull().to_numpy()

    assert bm_dataset.full_data.shape == bm_dataset.miss_data.shape
    assert (bm_dataset.full_data.columns == bm_dataset.miss_data.columns).all()

    bm_dataset._get_wgts()

    return bm_dataset


def ames_housing(
    n=3000, ci=True, mcar_prop=0.3, missing_mech: str = "linear"
) -> Dataset:
    """Ames housing prices (OpenML id=43952) with MAR/MNAR masking."""

    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=43952, as_frame=True, data_home=_get_cache_dir())
    frame = data.frame.copy()

    preferred_targets = ["SalePrice", "saleprice", "Sale_Price", "sale_price"]
    target_col = None

    if data.target is not None and data.target.name in frame.columns:
        target_col = data.target.name
    else:
        for cand in preferred_targets:
            if cand in frame.columns:
                target_col = cand
                break
    if target_col is None:
        num_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
        target_col = num_cols[0] if len(num_cols) > 0 else frame.columns[0]

    y_series = pd.to_numeric(frame[target_col], errors="coerce")
    if not np.isfinite(y_series).any():
        y_series = pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    else:
        y_series = y_series.fillna(y_series.median())

    frame = frame.drop(columns=[target_col])
    frame.insert(0, "y", y_series)

    n = min(n, frame.shape[0])
    idxs = np.random.choice(frame.shape[0], n, replace=False)
    compl = frame.iloc[idxs, :].reset_index(drop=True)

    ah_dataset = Dataset()
    ah_dataset.make(compl, y="y")
    ah_dataset.full_data = ah_dataset.miss_data.copy()

    mask_cols = [
        c for c in ah_dataset.full_data.columns if "Qual" in c or "GrLivArea" in c
    ]
    if len(mask_cols) == 0:
        mask_cols = ah_dataset.full_data.columns[1:6].tolist()

    ah_miss = ah_dataset.miss_data.copy()
    rng_u = np.random.rand(ah_miss.shape[0])

    qual_col = next(
        (c for c in compl.columns if "OverallQual" in c),
        ah_dataset.full_data.columns[1],
    )
    area_col = next(
        (c for c in compl.columns if "GrLivArea" in c),
        ah_dataset.full_data.columns[min(2, ah_dataset.full_data.shape[1] - 1)],
    )

    qual_numeric = pd.to_numeric(compl[qual_col], errors="coerce")
    if not np.isfinite(qual_numeric).any():
        qual_numeric = pd.Series(np.zeros(len(compl)), index=compl.index)

    area_numeric_full = pd.to_numeric(ah_dataset.full_data[area_col], errors="coerce")
    if not np.isfinite(area_numeric_full).any():
        area_numeric_full = pd.Series(
            np.zeros(len(ah_dataset.full_data)), index=ah_dataset.full_data.index
        )

    mech = missing_mech.lower()
    if mech == "linear":
        if ci:
            trigger = (qual_numeric > qual_numeric.median()).to_numpy()
        else:
            trigger = compl["y"].to_numpy() > np.median(compl["y"])

    elif mech == "xor":
        g1 = (qual_numeric > qual_numeric.median()).to_numpy()
        g2 = _to_binary_gate(
            area_numeric_full.fillna(area_numeric_full.median())
        ).to_numpy()
        base = g1 ^ g2
        if ci:
            trigger = base
        else:
            trigger = base ^ (compl["y"].to_numpy() > np.median(compl["y"]))

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    ah_miss.loc[miss_rows, mask_cols] = pd.NA

    for c in np.random.choice(
        a=ah_miss.shape[1] - 1,
        size=int(ah_miss.shape[1] * mcar_prop),
    ):
        ah_miss.iloc[
            np.random.choice(ah_miss.shape[0], int(ah_miss.shape[0] * 0.5)), c + 1
        ] = np.nan

    ah_dataset.miss_data = ah_miss
    ah_dataset.mask = ~ah_dataset.miss_data.isnull().to_numpy()

    assert ah_dataset.full_data.shape == ah_dataset.miss_data.shape
    assert (ah_dataset.full_data.columns == ah_dataset.miss_data.columns).all()

    ah_dataset._get_wgts()

    return ah_dataset


def give_me_some_credit(
    n=10000, ci=True, mcar_prop=0.3, missing_mech: str = "linear"
) -> Dataset:
    """Give Me Some Credit (OpenML) credit default with MAR/MNAR masking."""

    from sklearn.datasets import fetch_openml

    data = fetch_openml(
        "GiveMeSomeCredit", version=1, as_frame=True, data_home=_get_cache_dir()
    )
    frame = data.frame.copy()
    target_col = (
        "SeriousDlqin2yrs" if "SeriousDlqin2yrs" in frame.columns else data.target.name
    )
    frame = frame.rename(columns={target_col: "y"})
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")

    n = min(n, frame.shape[0])
    idxs = np.random.choice(frame.shape[0], n, replace=False)
    compl = frame.iloc[idxs, :].reset_index(drop=True)
    compl["y"] = compl["y"].fillna(0).astype(int)
    compl = pd.concat([compl["y"], compl.drop(columns=["y"])], axis=1)

    g_dataset = Dataset()
    g_dataset.make(compl, y="y")
    g_dataset.full_data = g_dataset.miss_data.copy()

    mask_cols = [
        c
        for c in g_dataset.full_data.columns
        if "RevolvingUtilization" in c or "DebtRatio" in c
    ]
    if len(mask_cols) == 0:
        mask_cols = g_dataset.full_data.columns[1:6].tolist()

    g_miss = g_dataset.miss_data.copy()
    rng_u = np.random.rand(g_miss.shape[0])

    util_col = next(
        (c for c in compl.columns if "RevolvingUtilization" in c),
        g_dataset.full_data.columns[1],
    )
    age_col = (
        "age"
        if "age" in compl.columns
        else g_dataset.full_data.columns[min(2, g_dataset.full_data.shape[1] - 1)]
    )

    mech = missing_mech.lower()
    if mech == "linear":
        if ci:
            trigger = (compl[util_col] > compl[util_col].median()).to_numpy()
        else:
            trigger = compl["y"].to_numpy() == 1

    elif mech == "xor":
        g1 = (compl[util_col] > compl[util_col].median()).to_numpy()
        age_numeric_full = pd.to_numeric(
            g_dataset.full_data[age_col], errors="coerce"
        ).fillna(0)
        g2 = _to_binary_gate(age_numeric_full).to_numpy()
        base = g1 ^ g2
        if ci:
            trigger = base
        else:
            trigger = base ^ (compl["y"].to_numpy() == 1)

    else:
        raise ValueError("missing_mech must be one of 'linear' or 'xor'")

    miss_rows = trigger & (rng_u < 0.9)
    g_miss.loc[miss_rows, mask_cols] = pd.NA

    for c in np.random.choice(
        a=g_miss.shape[1] - 1,
        size=int(g_miss.shape[1] * mcar_prop),
    ):
        g_miss.iloc[
            np.random.choice(g_miss.shape[0], int(g_miss.shape[0] * 0.5)), c + 1
        ] = np.nan

    g_dataset.miss_data = g_miss
    g_dataset.mask = ~g_dataset.miss_data.isnull().to_numpy()

    assert g_dataset.full_data.shape == g_dataset.miss_data.shape
    assert (g_dataset.full_data.columns == g_dataset.miss_data.columns).all()

    g_dataset._get_wgts()

    return g_dataset
