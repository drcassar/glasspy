"""This is the module for loading available data using GlassPy.

Currently the main source of GlassPy data is the SciGlass database.
The SciGlass database is available at https://github.com/epam/SciGlass
and is licensed under the ODC Open Database License (ODbL). A
text-only version of this database can be found at
https://github.com/drcassar/SciGlass. The data shipped with GlassPy is
the same as the data in the plain text fork.

Typical usage example:

  source = SciGlass()
  df = source.data

"""

import io
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests
from glasspy.chemistry.convert import to_element_array, wt_to_mol
from platformdirs import user_data_dir

from .translators import AtMol_translation, SciGK_translation


def _download_sciglass_data(path_dict):
    """Downloads the SciGlass database to your computer."""

    print("Downloading SciGlass database to your computer...")
    print("This is only required once and may take a few minutes.")

    record_id = "8287159"
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url, timeout=60)
    record_data = response.json()
    url = record_data["files"][0]["links"]["self"]

    download = requests.get(url, timeout=3600)
    zip_file = io.BytesIO(download.content)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for item in zip_ref.namelist():
            for name, path in path_dict.values():
                if name in item:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(item) as source_file:
                        with open(path, "wb") as target_file:
                            shutil.copyfileobj(source_file, target_file)

    print("Download completed!")


def _sciglass_path_dict():
    """Get the SciGlass file paths in your system."""

    data_dir = Path(user_data_dir("GlassPy"))

    path_dict = {
        "elements": ("AtMol", data_dir / "select_AtMol.csv.zip"),
        "properties": ("SciGK", data_dir / "select_SciGK.csv.zip"),
        "compounds": ("Gcomp", data_dir / "select_Gcomp.csv.zip"),
    }

    if not all(val[1].is_file() for val in path_dict.values()):
        _download_sciglass_data(path_dict)

    return path_dict


def sciglass_dbinfo():
    """Prints the SciGlass database information."""

    print()

    for key, value in SciGK_translation.items():
        name = value.get("rename", key)
        info = value.get("info", "")
        units = value.get("unit", None)
        units = f" ({units})" if units else ""
        meta = value.get("metadata", False)
        meta = " [metadata]" if meta else ""
        print(f"{name}: {info}{units}{meta}")


class SciGlass:
    """Loader of SciGlass data.

    Args:
      elements_cfg:
        Dictonary configuring how the `elements` information is collected. See
        docstring for `get_elements` method for more details.
      properties_cfg:
        Dictonary configuring how the `properties` information is collected. See
        docstring for `get_properties` method for more details.
      compounds_cfg:
        Dictonary configuring how the `compounds` information is collected. See
        docstring for `get_compounds` method for more details.
      autocleanup:
        If `True`, automatically delete columns of the final DataFrame that do
        not have any information (only zeros). Default value: True.
      metadata:
        If `True`, add the `metadata` information to the DataFrame. Default
        value: True.

    Attributes:
      data: DataFrame of the collected data.

    """

    def __init__(
        self,
        elements_cfg: dict | None = None,
        properties_cfg: dict | None = None,
        compounds_cfg: dict | None = None,
        autocleanup: bool = True,
        metadata: bool = True,
    ):
        path_dict = _sciglass_path_dict()

        if elements_cfg is None:
            elements_cfg = {
                "path": path_dict["elements"][1],
                "translate": AtMol_translation,
                "acceptable_sum_deviation": 1,
                "final_sum": 1,
            }

        if compounds_cfg is None:
            compounds_cfg = {
                "path": path_dict["compounds"][1],
                "acceptable_sum_deviation": 1,
                "final_sum": 1,
                "return_weight": False,
                "dropline": [
                    "",
                    "Al2O3+Fe2O3",
                    "MoO3+WO3",
                    "CaO+MgO",
                    "FeO+Fe2O3",
                    "Li2O+Na2O+K2O",
                    "Na2O+K2O",
                    "F2O-1",
                    "FemOn",
                    "HF+H2O",
                    "R2O",
                    "R2O3",
                    "R2O3",
                    "RO",
                    "RmOn",
                ],
            }

        if properties_cfg is None:
            properties_cfg = {
                "path": path_dict["properties"][1],
                "translate": SciGK_translation,
                "keep": self.available_properties(),
            }

        dfs = {}

        if properties_cfg:
            df = self.get_properties(**properties_cfg)
            dfs["property"] = df

        if elements_cfg:
            if "property" in dfs:
                df = self.get_elements(
                    IDs=dfs["property"].index, **elements_cfg
                )
            else:
                df = self.get_elements(**elements_cfg)
            dfs["elements"] = df

        if compounds_cfg:
            self.data = pd.concat(dfs, axis=1, join="inner")
            if len(self.data) > 0:
                df = self.get_compounds(IDs=self.data.index, **compounds_cfg)
            else:
                df = self.get_compounds(**compounds_cfg)
            dfs["compounds"] = df

        if metadata:
            metadata_cfg = {
                "path": path_dict["properties"][1],
                "translate": SciGK_translation,
                "keep": self.available_properties_metadata(),
            }

            df = self.get_properties(**metadata_cfg)
            dfs["metadata"] = df

            if "elements" in dfs:
                numelements = dfs["elements"].astype(bool).sum(axis=1)
                dfs["metadata"] = dfs["metadata"].assign(
                    NumberElements=numelements
                )

            if "compounds" in dfs:
                numcompounds = dfs["compounds"].astype(bool).sum(axis=1)
                dfs["metadata"] = dfs["metadata"].assign(
                    NumberCompounds=numcompounds
                )

            dfs["metadata"] = dfs["metadata"].convert_dtypes()

        # reordering
        dfs = {
            k: dfs[k]
            for k in ["elements", "compounds", "property", "metadata"]
            if k in dfs
        }

        self.data = pd.concat(dfs, axis=1, join="inner")

        if autocleanup:
            if "elements" in self.data.columns:
                self.remove_zero_sum_columns(scope="elements")

            if "compounds" in self.data.columns:
                self.remove_zero_sum_columns(scope="compounds")

    def get_properties(self, **kwargs):
        """Get properties information.

        Args:
          path : str
            String with the path to the database csv file.
          keep : list
            List of properties to keep in the final DataFrame.
          drop : list
            List of properties to remove from the final DataFrame.
          translate : dict
            Dictionary with the information on how to read and convert the
            properties. See variable `SciSK_translation` for an example.
          must_have_or: list
            List of properties that must be present in the final
            DataFrame (using OR logic)
          must_have_and: list
            List of properties that must be present in the final
            DataFrame (using AND logic)
          IDs : pd.Index
            IDs of the dataset to consider. Each glass in the SciGlass database
            has a glass number and a paper number. This ID used in GlassPy is an
            integer that merges both numbers

        """
        path_dict = _sciglass_path_dict()

        df = pd.read_csv(
            kwargs.get("path", path_dict["properties"][1]),
            sep="\t",
            low_memory=False,
        )
        df = df.assign(ID=lambda x: x.KOD * 100000000 + x.GLASNO)
        df = df.drop(["KOD", "GLASNO"], axis=1)
        translate = kwargs.get("translate", SciGK_translation)
        rename = {
            k: v["rename"] for k, v in translate.items() if "rename" in v
        }
        convert = {
            v.get("rename", k): v["convert"]
            for k, v in translate.items()
            if "convert" in v
        }

        df = self._process_df(
            df, rename=rename, convert=convert, origin="properties", **kwargs
        )

        return df

    def get_elements(self, **kwargs):
        """Get elemental atomic fraction information.

        Args:
          path : str
            String with the path to the database csv file.
          keep : list
            List of elements to keep in the final DataFrame.
          drop : list
            List of elements to remove from the final DataFrame.
          translate : dict
            Dictionary with the information on how to read and convert the
            elements. See variable `AtMol_translation` for an example.
          acceptable_sum_deviation : positive int or float
            The sum of all atomic fractions should be 100%.  However, due to
            float point errors or rounding errors, this sum will not be exactly
            100%. This argument controls the acceptable deviation of this sum in
            %. A value of 1 means that the sum of all atomic fractions can be
            between 99 and 101. All examples that are not within this range are
            discarted.
          final_sum : positive int or float
            The final sum of all atomic fractions is normalized to this value.
            Usual values are 1 if you want atomic fractions or 100 if you want
            atomic percentages.
          must_have_or: list
            List of elements that must be present in the final
            DataFrame (using OR logic)
          must_have_and: list
            List of elements that must be present in the final
            DataFrame (using AND logic)
          IDs : pd.Index
            IDs of the dataset to consider. Each glass in the SciGlass database
            has a glass number and a paper number. This ID used in GlassPy is an
            integer that merges both numbers

        """
        path_dict = _sciglass_path_dict()

        df = pd.read_csv(
            kwargs.get("path", path_dict["elements"][1]),
            sep="\t",
            low_memory=False,
        )
        df = df.assign(ID=lambda x: x.Kod * 100000000 + x.GlasNo)
        df = df.drop(["Kod", "GlasNo"], axis=1)
        translate = kwargs.get("translate", AtMol_translation)
        df = df.set_index("ID", drop=False)
        if "IDs" in kwargs:
            idx = df.index.intersection(kwargs["IDs"])
            df = df.loc[idx]
        df = self._process_df(
            df, rename=translate, origin="elements", **kwargs
        )
        return df

    def get_compounds(self, **kwargs):
        """Get compound information.

        Args:
          path : str
            String with the path to the database csv file.
          keep : list
            List of compounds to keep in the final DataFrame.
          drop : list
            List of compounds to remove from the final DataFrame.
          acceptable_sum_deviation : positive int or float
            The sum of all compound fractions should be 100%.  However, due to
            float point errors or rounding errors, this sum will not be exactly
            100%. This argument controls the acceptable deviation of this sum in
            %. A value of 1 means that the sum of all compound fractions can be
            between 99 and 101. All examples that are not within this range are
            discarted.
          final_sum : positive int or float
            The final sum of all compound fractions is normalized to this value.
            Usual values are 1 if you want compound fractions or 100 if you want
            compound percentages.
          return_weight : bool
            If `True`, the chemical information stored in the DataFrame will be
            in weight%. Otherwise it will be in mol%.
          must_have_or: list
            List of compounds that must be present in the final
            DataFrame (using OR logic)
          must_have_and: list
            List of compounds that must be present in the final
            DataFrame (using AND logic)
          IDs : pd.Index
            IDs of the dataset to consider. Each glass in the SciGlass database
            has a glass number and a paper number. This ID used in GlassPy is an
            integer that merges both numbers

        """
        path_dict = _sciglass_path_dict()

        df = pd.read_csv(
            kwargs.get("path", path_dict["compounds"][1]),
            sep="\t",
            low_memory=False,
        )
        df = df.assign(ID=lambda x: x.Kod * 100000000 + x.GlasNo)
        df = df.drop(["Kod", "GlasNo"], axis=1)
        df = df.set_index("ID", drop=True)

        if "IDs" in kwargs:
            idx = df.index.intersection(kwargs["IDs"])
            df = df.loc[idx]

        df = df["Composition"].str.slice(start=1, stop=-1)
        df = df.str.split("\x7f")

        add = 0 if kwargs.get("return_weight", False) else 1

        compound_list = []
        for row in df.values:
            compound_list.append(
                {
                    row[n * 4]: float(row[(n * 4 + 2 + add)])
                    for n in range(len(row) // 4)
                }
            )

        data = (
            pd.DataFrame(compound_list)
            .dropna(axis=1, how="all")
            .assign(ID=df.index)
            .dropna(axis=0, how="all")
            .fillna(0)
        )

        df = self._process_df(data, origin="compounds", **kwargs)
        return df

    @staticmethod
    def _process_df(df, **kwargs):
        """Function to process the DataFrame."""

        df = df.drop_duplicates("ID", keep=False)
        df = df.set_index("ID", drop=True)

        if "rename" in kwargs:
            df = df.rename(columns=kwargs["rename"])
        if kwargs["origin"] == "properties":
            if "must_have_or" in kwargs:
                strings = [
                    f"{prop}.notna()" for prop in kwargs["must_have_or"]
                ]
                query = " or ".join(strings)
                df = df.query(query)
            if "must_have_and" in kwargs:
                strings = [
                    f"{prop}.notna()" for prop in kwargs["must_have_and"]
                ]
                query = " and ".join(strings)
                df = df.query(query)
        else:
            if "must_have_or" in kwargs:
                strings = [f"{comp} > 0" for comp in kwargs["must_have_or"]]
                query = " or ".join(strings)
                df = df.query(query)
            if "must_have_and" in kwargs:
                strings = [f"{comp} > 0" for comp in kwargs["must_have_and"]]
                query = " and ".join(strings)
                df = df.query(query)
        if "keep" in kwargs:
            df = df.reindex(kwargs["keep"], axis=1)
        if "drop" in kwargs:
            df = df.drop(kwargs["drop"], axis=1, errors="ignore")
        if "drop_compound_with_element" in kwargs:
            drop_cols = []
            for col in df.columns:
                for el in kwargs["drop_compound_with_element"]:
                    if el in col:
                        df = df.loc[df[col] == 0]
                        drop_cols.append(col)
                        # no need to check other elements, the column is gone
                        break
            df = df.drop(drop_cols, axis=1, errors="ignore")
        if "dropline" in kwargs:
            for col in kwargs["dropline"]:
                if col in df.columns:
                    df = df.loc[df[col] == 0]
            df = df.drop(kwargs["dropline"], axis=1, errors="ignore")
        if "convert" in kwargs:
            for k, v in kwargs["convert"].items():
                if k in df.columns:
                    df[k] = df[k].apply(v)
        if "acceptable_sum_deviation" in kwargs:
            diff = df.sum(axis=1) - 100
            thres = kwargs["acceptable_sum_deviation"]
            df = df.loc[diff.between(-thres, thres)]
        if "final_sum" in kwargs:
            df = df.divide(df.sum(axis=1), axis=0) * kwargs["final_sum"]

        df = df.dropna(axis=0, how="all")
        df = df.dropna(axis=1, how="all")

        return df

    def remove_zero_sum_columns(self, scope="compounds"):
        """Removes all columns that have zero sum from the `data` attribute."""

        zero_sum_cols = self.data[scope].sum(axis=0) == 0
        drop_cols = self.data[scope].columns[zero_sum_cols].tolist()
        drop_cols = [(scope, el) for el in drop_cols]
        self.data.drop(drop_cols, axis=1, inplace=True)

    def remove_duplicate_composition(
        self, scope="elements", decimals=3, aggregator="median"
    ):
        """Remove duplicate compositions from the `data` attribute.

        Note that the original ID and the metadata are lost upon this operation.

        """
        assert scope in ["elements", "compounds"]
        assert "property" in self.data.columns.levels[0]
        assert scope in self.data.columns.levels[0]

        comp_cols = self.data[scope].columns.to_list()
        prop_cols = self.data["property"].columns.to_list()
        df = self.data[[scope, "property"]].droplevel(0, axis=1)
        df[comp_cols] = df[comp_cols].round(decimals)
        grouped = df.groupby(comp_cols, sort=False)
        df = getattr(grouped, aggregator)().reset_index()
        df = {scope: df[comp_cols], "property": df[prop_cols]}
        self.data = pd.concat(df, axis=1, join="inner")

    def elements_from_compounds(self, final_sum=1, compounds_in_weight=False):
        """Create atomic fraction information from compound information.

        Args:
          final_sum : positive int or float
            The final sum of all atomic fractions is normalized to this value.
            Usual values are 1 if you want atomic fractions or 100 if you want
            atomic percentages. Default value is 1.
          compounds_in_weight : bool
            If `True`, then assume that the compounds fractions are in weight%,
            otherwise assume that the compounds fractions are in mol%. Default
            value is `False`.

        """
        assert "compounds" in self.data.columns.levels[0]
        assert "elements" not in self.data.columns.levels[0]

        chemarray = to_element_array(
            self.data["compounds"], rescale_to_sum=final_sum
        )

        if compounds_in_weight:
            chemarray = wt_to_mol(chemarray, chemarray.cols, final_sum)

        el_df = pd.DataFrame(
            chemarray,
            columns=chemarray.cols,
            index=self.data["compounds"].index,
        )
        dfs = {k: self.data[k] for k in self.data.columns.levels[0]}
        dfs["elements"] = el_df

        # reordering
        dfs = {
            k: dfs[k]
            for k in ["elements", "compounds", "property", "metadata"]
            if k in dfs.keys()
        }

        self.data = pd.concat(dfs, axis=1, join="inner")

    @staticmethod
    def available_properties():
        """Returns a list of available properties."""

        metadata = [
            v.get("rename", k)
            for k, v in SciGK_translation.items()
            if v.get("metadata", False)
        ]

        return [
            SciGK_translation[k].get("rename", k)
            for k in SciGK_translation
            if SciGK_translation[k].get("rename", k) not in metadata
        ]

    @staticmethod
    def available_properties_metadata():
        """Returns a list of available properties metadata."""

        return [
            v.get("rename", k)
            for k, v in SciGK_translation.items()
            if v.get("metadata", False)
        ]

    @staticmethod
    def database_info():
        """Prints database information."""
        sciglass_dbinfo()
