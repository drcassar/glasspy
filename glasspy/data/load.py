import pandas as pd
import numpy as np
import os

from glasspy.chemistry.convert import to_element_array, wt_to_mol
from .translators import AtMol_translation, SciGK_translation

__cur_path = os.path.dirname(__file__)
_ELEMENTS_PATH = os.path.join(__cur_path, "datafiles/select_AtMol.csv.zip")
_PROPERTIES_PATH = os.path.join(__cur_path, "datafiles/select_SciGK.csv.zip")
_COMPOUNDS_PATH = os.path.join(__cur_path, "datafiles/select_Gcomp.csv.zip")


class SciGlass:
    """Loader of SciGlass data."""

    def __init__(
        self,
        elements_cfg={},
        properties_cfg={},
        compounds_cfg={},
        autocleanup=True,
        metadata=True,
    ):

        # default behavior is returning everything if no config is given
        if (not elements_cfg) and (not properties_cfg) and (not compounds_cfg):
            elements_cfg = {
                "path": _ELEMENTS_PATH,
                "translate": AtMol_translation,
                "acceptable_sum_deviation": 1,
                "final_sum": 1,
            }

            compounds_cfg = {
                "path": _COMPOUNDS_PATH,
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

            properties_cfg = {
                "path": _PROPERTIES_PATH,
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
                "path": _PROPERTIES_PATH,
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
            if k in dfs.keys()
        }

        self.data = pd.concat(dfs, axis=1, join="inner")

        if autocleanup:
            if "elements" in self.data.columns:
                self.remove_zero_sum_columns(scope="elements")

            if "compounds" in self.data.columns:
                self.remove_zero_sum_columns(scope="compounds")

    def get_properties(self, **kwargs):
        """Get elemental atomic fraction information.

        Args:
          path
          keep
          drop
          translate
          IDs

        """
        df = pd.read_csv(
            kwargs.get("path", _PROPERTIES_PATH), sep="\t", low_memory=False
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
        df = self.process_df(df, rename=rename, convert=convert, **kwargs)
        return df

    def get_elements(self, **kwargs):
        """Get elemental atomic fraction information.

        Args:
          path
          keep
          drop
          translate
          acceptable_sum_deviation
          final_sum
          IDs

        """
        df = pd.read_csv(
            kwargs.get("path", _ELEMENTS_PATH), sep="\t", low_memory=False
        )
        df = df.assign(ID=lambda x: x.Kod * 100000000 + x.GlasNo)
        df = df.drop(["Kod", "GlasNo"], axis=1)
        translate = kwargs.get("translate", AtMol_translation)
        df = df.set_index("ID", drop=False)
        if "IDs" in kwargs:
            idx = df.index.intersection(kwargs["IDs"])
            df = df.loc[idx]
        df = self.process_df(df, rename=translate, **kwargs)
        return df

    def get_compounds(self, **kwargs):
        """Get elemental atomic fraction information.

        Args:
          path
          keep
          rename
          acceptable_sum_deviation
          final_sum
          return_weight
          IDs

        """
        df = pd.read_csv(
            kwargs.get("path", _COMPOUNDS_PATH), sep="\t", low_memory=False
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
        df = self.process_df(data, **kwargs)
        return df

    @staticmethod
    def process_df(df, **kwargs):
        df = df.drop_duplicates("ID", keep=False)
        df = df.set_index("ID", drop=True)

        if "rename" in kwargs:
            df = df.rename(columns=kwargs["rename"])
        if "keep" in kwargs:
            df = df.reindex(kwargs["keep"], axis=1)
        if "must_have_or" in kwargs:
            strings = [f"{el} > 0" for el in kwargs["must_have_or"]]
            query = " or ".join(strings)
            df = df.query(query)
        if "must_have_and" in kwargs:
            strings = [f"{el} > 0" for el in kwargs["must_have_and"]]
            query = " and ".join(strings)
            df = df.query(query)
        if "drop" in kwargs:
            df = df.drop(kwargs["drop"], axis=1, errors="ignore")
        if "drop_compound_with_element" in kwargs:
            drop_cols = []
            for col in df.columns:
                for el in kwargs["drop_compound_with_element"]:
                    if el in col:
                        df = df.loc[df[col] == 0]
                        drop_cols.append(col)
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
        zero_sum_cols = self.data[scope].sum(axis=0) == 0
        drop_cols = self.data[scope].columns[zero_sum_cols].tolist()
        drop_cols = [(scope, el) for el in drop_cols]
        self.data.drop(drop_cols, axis=1, inplace=True)

    def remove_duplicate_composition(
        self, scope="elements", decimals=3, aggregator="median"
    ):
        """Remove duplicate compositions

        Note that the original ID and metadata are lost upon this operation.

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
        metadata = [
            SciGK_translation[k].get("rename", k)
            for k in SciGK_translation
            if SciGK_translation[k].get("metadata", False)
        ]

        return [
            SciGK_translation[k].get("rename", k)
            for k in SciGK_translation
            if SciGK_translation[k].get("rename", k) not in metadata
        ]

    @staticmethod
    def available_properties_metadata():
        return [
            SciGK_translation[k].get("rename", k)
            for k in SciGK_translation
            if SciGK_translation[k].get("metadata", False)
        ]
