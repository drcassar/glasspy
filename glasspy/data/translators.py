AtMol_translation = {
    "1": "H", "2": "He", "3": "Li", "4": "Be", "5": "B", "6": "C", "7": "N",
    "8": "O", "9": "F", "10": "Ne", "11": "Na", "12": "Mg", "13": "Al", "14":
    "Si", "15": "P", "16": "S", "17": "Cl", "18": "Ar", "19": "K", "20": "Ca",
    "21": "Sc", "22": "Ti", "23": "V", "24": "Cr", "25": "Mn", "26": "Fe", "27":
    "Co", "28": "Ni", "29": "Cu", "30": "Zn", "31": "Ga", "32": "Ge", "33":
    "As", "34": "Se", "35": "Br", "36": "Kr", "37": "Rb", "38": "Sr", "39": "Y",
    "40": "Zr", "41": "Nb", "42": "Mo", "43": "Tc", "44": "Ru", "45": "Rh",
    "46": "Pd", "47": "Ag", "48": "Cd", "49": "In", "50": "Sn", "51": "Sb",
    "52": "Te", "53": "I", "54": "Xe", "55": "Cs", "56": "Ba", "57": "La", "58":
    "Ce", "59": "Pr", "60": "Nd", "61": "Pm", "62": "Sm", "63": "Eu", "64":
    "Gd", "65": "Tb", "66": "Dy", "67": "Ho", "68": "Er", "69": "Tm", "70":
    "Yb", "71": "Lu", "72": "Hf", "73": "Ta", "74": "W", "75": "Re", "76": "Os",
    "77": "Ir", "78": "Pt", "79": "Au", "80": "Hg", "81": "Tl", "82": "Pb",
    "83": "Bi", "84": "Po", "85": "At", "86": "Rn", "87": "Fr", "88": "Ra",
    "89": "Ac", "90": "Th", "91": "Pa", "92": "U", "93": "Np", "94": "Pu",
}

SciGK_translation = {

    # Metadata
    "Analysis": {
        "info": "Indicates if the glass composition was obtained by"
                " chemical analysis",
        "rename": "ChemicalAnalysis",
        "convert": lambda x: True if x == "a" else False,
        "metadata": True,
    },
    "Author": {
        "info": "First author of the publication",
        "metadata": True,
    },
    "Year": {
        "info": "Year of the publication",
        "metadata": True,
    },

    # Viscosity
    "T1": {
        "info": "Temperature where viscosity is 1 Pa.s",
        "rename": "T0",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T2": {
        "info": "Temperature where viscosity is 10 Pa.s",
        "rename": "T1",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T3": {
        "info": "Temperature where viscosity is 100 Pa.s",
        "rename": "T2",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T4": {
        "info": "Temperature where viscosity is 1000 Pa.s",
        "rename": "T3",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T5": {
        "info": "Temperature where viscosity is 10000 Pa.s",
        "rename": "T4",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T6": {
        "info": "Temperature where viscosity is 100000 Pa.s",
        "rename": "T5",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T7": {
        "info": "Temperature where viscosity is 1000000 Pa.s",
        "rename": "T6",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T8": {
        "info": "Temperature where viscosity is 10000000 Pa.s",
        "rename": "T7",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T9": {
        "info": "Temperature where viscosity is 100000000 Pa.s",
        "rename": "T8",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T10": {
        "info": "Temperature where viscosity is 1000000000 Pa.s",
        "rename": "T9",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T11": {
        "info": "Temperature where viscosity is 10000000000 Pa.s",
        "rename": "T10",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T12": {
        "info": "Temperature where viscosity is 100000000000 Pa.s",
        "rename": "T11",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "T13": {
        "info": "Temperature where viscosity is 1000000000000 Pa.s",
        "rename": "T12",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "V500": {
        "info": "Viscosity at 773 K",
        "rename": "Viscosity773K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V600": {
        "info": "Viscosity at 873 K",
        "rename": "Viscosity873K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V700": {
        "info": "Viscosity at 973 K",
        "rename": "Viscosity973K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V800": {
        "info": "Viscosity at 1073 K",
        "rename": "Viscosity1073K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V900": {
        "info": "Viscosity at 1173 K",
        "rename": "Viscosity1173K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V1000": {
        "info": "Viscosity at 1273 K",
        "rename": "Viscosity1273K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V1100": {
        "info": "Viscosity at 1373 K",
        "rename": "Viscosity1373K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V1200": {
        "info": "Viscosity at 1473 K",
        "rename": "Viscosity1473K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V1300": {
        "info": "Viscosity at 1573 K",
        "rename": "Viscosity1573K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V1400": {
        "info": "Viscosity at 1673 K",
        "rename": "Viscosity1673K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V1500": {
        "info": "Viscosity at 1773 K",
        "rename": "Viscosity1773K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V1600": {
        "info": "Viscosity at 1873 K",
        "rename": "Viscosity1873K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V1800": {
        "info": "Viscosity at 2073 K",
        "rename": "Viscosity2073K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V2000": {
        "info": "Viscosity at 2273 K",
        "rename": "Viscosity2273K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    "V2200": {
        "info": "Viscosity at 2473 K",
        "rename": "Viscosity2473K",
        "convert": lambda x: 10 ** (x - 1),
        "unit": "Pa.s",
    },
    # Temperatures
    "TG": {
        "info": "Glass transition temperature",
        "rename": "Tg",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "Tm": {
        "info": "Melting temperature",
        "rename": "Tmelt",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "TLiq": {
        "info": "Liquidus temperature",
        "rename": "Tliquidus",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "LPT": {
        "info": "Littletons softening temperature",
        "rename": "TLittletons",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "ANPT": {
        "info": "Annealing point",
        "rename": "TAnnealing",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "SPT": {
        "info": "Strain point",
        "rename": "Tstrain",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "Tsoft": {
        "info": "Softening point",
        "rename": "Tsoft",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "Mg": {
        "info": "Dilatometric softening temperature",
        "rename": "TdilatometricSoftening",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },

    # Optical
    "NUD300": {
        "info": "Abbe's number",
        "rename": "AbbeNum",
    },
    "ND300": {
        "info": "Refractive index",
        "rename": "RefractiveIndex",
    },
    "nd300low": {
        "info": "Refractive index measured at a wavelenght between 0.6 and 1 "
                "micron at 293 K",
        "rename": "RefractiveIndexLow",
    },
    "nd300hi": {
        "info": "Refractive index measured at a wavelenght greater than 1 "
                "micron at 293 K",
        "rename": "RefractiveIndexHigh",
    },
    "DNFC300": {
        "info": "Mean dispersion (nF - nC)",
        "rename": "MeanDispersion",
        "convert": lambda x: x * 1e-4,
    },

    # Electrical and dielectrical
    "EPS730": {
        "info": "Relative permittivity at ambient temperature anf frequency of"
                "1 MHz (or the nearest frequency in the range of 0.01 MHz to "
                "10 MHz)",
        "rename": "Permittivity",
    },
    "TGD730": {
        "info": "Tangent of loss angle",
        "rename": "TangentOfLossAngle",
        "convert": lambda x: x * 1e-4,
    },
    "TK100C": {
        "info": "Temperature where the specific electrical resistivity is 1"
                "MOhm.m",
        "rename": "TresistivityIs1MOhm.m",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "RO20": {
        "info": "Specific electrical resistivity measured at 273 K",
        "rename": "Resistivity273K",
        "convert": lambda x: 10 ** (x - 4),
        "unit": "Ohm.m",
    },
    "RO100": {
        "info": "Specific electrical resistivity measured at 373 K",
        "rename": "Resistivity373K",
        "convert": lambda x: 10 ** (x - 4),
        "unit": "Ohm.m",
    },
    "RO150": {
        "info": "Specific electrical resistivity measured at 423 K",
        "rename": "Resistivity423K",
        "convert": lambda x: 10 ** (x - 4),
        "unit": "Ohm.m",
    },
    "RO300": {
        "info": "Specific electrical resistivity measured at 573 K",
        "rename": "Resistivity573K",
        "convert": lambda x: 10 ** (x - 4),
        "unit": "Ohm.m",
    },
    "ro800": {
        "info": "Specific electrical resistivity measured at 1073 K",
        "rename": "Resistivity1073K",
        "convert": lambda x: 10 ** (x - 4),
        "unit": "Ohm.m",
    },
    "ro1000": {
        "info": "Specific electrical resistivity measured at 1273 K",
        "rename": "Resistivity1273K",
        "convert": lambda x: 10 ** (x - 4),
        "unit": "Ohm.m",
    },
    "ro1200": {
        "info": "Specific electrical resistivity measured at 1473 K",
        "rename": "Resistivity1473K",
        "convert": lambda x: 10 ** (x - 4),
        "unit": "Ohm.m",
    },
    "ro1400": {
        "info": "Specific electrical resistivity measured at 1673 K",
        "rename": "Resistivity1673K",
        "convert": lambda x: 10 ** (x - 4),
        "unit": "Ohm.m",
    },

    # Mechanical
    "MOD_UNG": {
        "info": "Young's Modulus",
        "rename": "YoungModulus",
        "unit": "GPa",
    },
    "MOD_SDV": {
        "info": "Shear Modulus",
        "rename": "ShearModulus",
        "unit": "GPa",
    },
    "MIKROTV": {
        "info": "Microhardness measured by Knoop or Vickers indentation",
        "rename": "Microhardness",
        "unit": "GPa",
    },
    "pois": {
        "info": "Poisson's ratio",
        "rename": "PoissonRatio",
    },

    # Density
    "DENSITY": {
        "info": "Density measured at 293 K",
        "rename": "Density293K",
        "unit": "g/cm3",
    },
    "dens800": {
        "info": "Density measured at 1073 K",
        "rename": "Density1073K",
        "unit": "g/cm3",
    },
    "dens1000": {
        "info": "Density measured at 1273 K",
        "rename": "Density1273K",
        "unit": "g/cm3",
    },
    "dens1200": {
        "info": "Density measured at 1473 K",
        "rename": "Density1473K",
        "unit": "g/cm3",
    },
    "dens1400": {
        "info": "Density measured at 1673 K",
        "rename": "Density1673K",
        "unit": "g/cm3",
    },

    # Thermal
    "cond220": {
        "info": "Thermal conductivity",
        "rename": "ThermalConductivity",
        "unit": "W/(m.K)",
    },
    "RTSH180": {
        "info": "Thermal shock resistance",
        "rename": "ThermalShockRes",
        "unit": "K",
    },
    "ANY_TEC": {
        "info": "Linear coefficient of thermal expansion measured below the "
                "glass transition temperature",
        "rename": "CTEbelowTg",
        "convert": lambda x: x * 1e-7,
        "unit": "1/K",
    },
    "TEC55": {
        "info": "Linear coefficient of thermal expansion measured at 328 +/- 10 K",
        "rename": "CTE328K",
        "convert": lambda x: x * 1e-7,
        "unit": "1/K",
    },
    "TEC100": {
        "info": "Linear coefficient of thermal expansion measured at 373 +/- 10 K",
        "rename": "CTE373K",
        "convert": lambda x: x * 1e-7,
        "unit": "1/K",
    },
    "TEC160": {
        "info": "Linear coefficient of thermal expansion measured at 433 +/- 10 K",
        "rename": "CTE433K",
        "convert": lambda x: x * 1e-7,
        "unit": "1/K",
    },
    "TEC210": {
        "info": "Linear coefficient of thermal expansion measured at 483 +/- 10 K",
        "rename": "CTE483K",
        "convert": lambda x: x * 1e-7,
        "unit": "1/K",
    },
    "TEC350": {
        "info": "Linear coefficient of thermal expansion measured at 623 +/- 10 K",
        "rename": "CTE623K",
        "convert": lambda x: x * 1e-7,
        "unit": "1/K",
    },
    "cp20": {
        "info": "Heat capacity at constant pressure measured at 293 K",
        "rename": "Cp293K",
        "unit": "J/(kg.K)",
    },
    "cp200": {
        "info": "Heat capacity at constant pressure measured at 473 K",
        "rename": "Cp473K",
        "unit": "J/(kg.K)",
    },
    "cp400": {
        "info": "Heat capacity at constant pressure measured at 673 K",
        "rename": "Cp673K",
        "unit": "J/(kg.K)",
    },
    "cp800": {
        "info": "Heat capacity at constant pressure measured at 1073 K",
        "rename": "Cp1073K",
        "unit": "J/(kg.K)",
    },
    "cp1000": {
        "info": "Heat capacity at constant pressure measured at 1273 K",
        "rename": "Cp1273K",
        "unit": "J/(kg.K)",
    },
    "cp1200": {
        "info": "Heat capacity at constant pressure measured at 1473 K",
        "rename": "Cp1473K",
        "unit": "J/(kg.K)",
    },
    "cp1400": {
        "info": "Heat capacity at constant pressure measured at 1673 K",
        "rename": "Cp1673K",
        "unit": "J/(kg.K)",
    },

    # Crystallization
    "Tn": {
        "info": "Nucleation temperature",
        "rename": "NucleationTemperature",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "Io": {
        "info": "Crystal nucleation rate",
        "rename": "NucleationRate",
        "convert": lambda x: x * 1e6,
        "unit": "1/(s.m3)",
    },
    "Tmax": {
        "info": "Temperature of maximum crystal growth velocity",
        "rename": "TMaxGrowthVelocity",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "Vmax": {
        "info": "Maximum crystal growth velocity",
        "rename": "MaxGrowthVelocity",
        "convert": lambda x: x / 100,
        "unit": "m/s",
    },
    "tcr": {
        "info": "DTA temperature of crystallization peak",
        "rename": "CrystallizationPeak",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },
    "tx": {
        "info": "DTA temperature of crystallization onset",
        "rename": "CrystallizationOnset",
        "convert": lambda x: x + 273.15,
        "unit": "K",
    },

    # Surface
    "any_sut": {
        "info": "Surface tension above the glass transition temperature",
        "rename": "SurfaceTensionAboveTg",
        "convert": lambda x: x / 1000,
        "unit": "J/m2",
    },
    "SUT900": {
        "info": "Surface tension at 1173 K",
        "rename": "SurfaceTension1173K",
        "convert": lambda x: x / 1000,
        "unit": "J/m2",
    },
    "SUT1200": {
        "info": "Surface tension at 1473 K",
        "rename": "SurfaceTension1473K",
        "convert": lambda x: x / 1000,
        "unit": "J/m2",
    },
    "SUT1300": {
        "info": "Surface tension at 1573 K",
        "rename": "SurfaceTension1573K",
        "convert": lambda x: x / 1000,
        "unit": "J/m2",
    },
    "SUT1400": {
        "info": "Surface tension at 1673 K",
        "rename": "SurfaceTension1673K",
        "convert": lambda x: x / 1000,
        "unit": "J/m2",
    },
}
