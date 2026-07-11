"""Global configurations and typed profiles for the B2B energy simulator.

This module defines the quantitative scales, seasonal consumption curves, and
industry archetypes that govern the synthetic business data generation pipeline.
It leverages strict domain vocabularies via typing literals and dictionary schemas
to enforce structural invariants during the simulation and seeding processes.

"""

from typing import Final, Literal, TypedDict

MONTHS_PER_YEAR: Final = 12

EnergyLevel = Literal["very_low", "low", "medium", "high", "very_high"]
CompanySize = Literal["small", "medium", "large"]
SeasonalityProfile = Literal["flat", "summer_peak", "winter_heavy", "summer_shutdown"]

SectorName = Literal[
    "manufacturing",
    "chemical_heavy_industry",
    "food_beverage",
    "services",
    "commerce",
    "hospitality",
    "healthcare",
    "agriculture",
    "construction",
    "transportation",
    "utilities",
    "tech",
]


class SeasonalityConfig(TypedDict):
    """Defines the seasonal consumption profiles for energy vectors.

    Attributes:
        electricity (SeasonalityProfile): The curve governing power consumption.
        gas (SeasonalityProfile): The curve governing thermal gas consumption.

    """

    electricity: SeasonalityProfile
    gas: SeasonalityProfile


class ContractMixConfig(TypedDict):
    """Represents the operational distribution weights for corporate supply contracts.

    Attributes:
        electricity_only (EnergyLevel): Probability or density of electricity contracts.
        gas_only (EnergyLevel): Probability or density of standalone gas contracts.
        dual_fuel (EnergyLevel): Probability or density of combined utility contracts.

    """

    electricity_only: EnergyLevel
    gas_only: EnergyLevel
    dual_fuel: EnergyLevel


class SectorProfile(TypedDict):
    """Configuration schema for an industrial sector's behavioral matrix.

    This schema encapsulates the entire operational baseline required to simulate
    realistic B2B corporate entities, encompassing energy weights, operational
    seasonality, financial volatility, and structural biases.

    Attributes:
        energy_intensity (EnergyLevel): Global volume baseline of energy consumption.
        electricity_weight (EnergyLevel): Proportional scale for power operations.
        gas_weight (EnergyLevel): Proportional scale for thermal operations.
        seasonality (SeasonalityConfig): Nested curves for monthly usage variations.
        revenue_volatility (EnergyLevel): Intensity of financial cash-flow shifts.
        digitalization_bias (EnergyLevel): Adaptability scale for automated systems.
        support_bias (EnergyLevel): Probability index of customer interaction events.
        contract_mix (ContractMixConfig): Distribution setup for supply models.

    """

    energy_intensity: EnergyLevel
    electricity_weight: EnergyLevel
    gas_weight: EnergyLevel
    seasonality: SeasonalityConfig
    revenue_volatility: EnergyLevel
    digitalization_bias: EnergyLevel
    support_bias: EnergyLevel
    contract_mix: ContractMixConfig


ENERGY_INTENSITY_SCALE: dict[EnergyLevel, float] = {
    "very_low": 0.1,
    "low": 0.5,
    "medium": 1.0,
    "high": 3.0,
    "very_high": 10.0,
}

COMPANY_SIZE_SCALE: dict[CompanySize, float] = {
    "small": 1.0,
    "medium": 4.0,
    "large": 15.0,
}

EVENT_RATE_SCALE: dict[EnergyLevel, float] = {
    "very_low": 0.2,
    "low": 0.6,
    "medium": 1.0,
    "high": 1.8,
    "very_high": 3.5,
}

SEASONALITY_PROFILE_CURVES: dict[SeasonalityProfile, tuple[float, ...]] = {
    "flat": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "summer_peak": (0.8, 0.8, 0.8, 0.9, 1.0, 1.3, 1.4, 1.3, 1.1, 0.9, 0.8, 0.9),
    "winter_heavy": (
        1.72,
        1.64,
        1.31,
        0.87,
        0.55,
        0.44,
        0.44,
        0.44,
        0.55,
        0.98,
        1.42,
        1.64,
    ),
    "summer_shutdown": (
        1.07,
        1.07,
        1.07,
        1.07,
        1.07,
        1.07,
        1.07,
        0.20,
        1.08,
        1.08,
        1.08,
        1.07,
    ),
}


SECTOR_PROFILES: dict[SectorName, SectorProfile] = {
    "manufacturing": {
        "energy_intensity": "medium",
        "electricity_weight": "high",
        "gas_weight": "medium",
        "seasonality": {"electricity": "summer_shutdown", "gas": "winter_heavy"},
        "revenue_volatility": "medium",
        "digitalization_bias": "medium",
        "support_bias": "medium",
        "contract_mix": {
            "electricity_only": "low",
            "gas_only": "very_low",
            "dual_fuel": "high",
        },
    },
    "chemical_heavy_industry": {
        "energy_intensity": "very_high",
        "electricity_weight": "very_high",
        "gas_weight": "very_high",
        "seasonality": {"electricity": "flat", "gas": "flat"},
        "revenue_volatility": "medium",
        "digitalization_bias": "medium",
        "support_bias": "high",
        "contract_mix": {
            "electricity_only": "medium",
            "gas_only": "low",
            "dual_fuel": "high",
        },
    },
    "food_beverage": {
        "energy_intensity": "high",
        "electricity_weight": "high",
        "gas_weight": "medium",
        "seasonality": {"electricity": "flat", "gas": "flat"},
        "revenue_volatility": "low",
        "digitalization_bias": "medium",
        "support_bias": "medium",
        "contract_mix": {
            "electricity_only": "high",
            "gas_only": "very_low",
            "dual_fuel": "medium",
        },
    },
    "services": {
        "energy_intensity": "very_low",
        "electricity_weight": "high",
        "gas_weight": "low",
        "seasonality": {"electricity": "summer_peak", "gas": "winter_heavy"},
        "revenue_volatility": "low",
        "digitalization_bias": "high",
        "support_bias": "medium",
        "contract_mix": {
            "electricity_only": "high",
            "gas_only": "very_low",
            "dual_fuel": "low",
        },
    },
    "commerce": {
        "energy_intensity": "low",
        "electricity_weight": "high",
        "gas_weight": "low",
        "seasonality": {"electricity": "summer_peak", "gas": "winter_heavy"},
        "revenue_volatility": "medium",
        "digitalization_bias": "medium",
        "support_bias": "medium",
        "contract_mix": {
            "electricity_only": "high",
            "gas_only": "low",
            "dual_fuel": "low",
        },
    },
    "hospitality": {
        "energy_intensity": "high",
        "electricity_weight": "high",
        "gas_weight": "high",
        "seasonality": {"electricity": "summer_peak", "gas": "winter_heavy"},
        "revenue_volatility": "high",
        "digitalization_bias": "medium",
        "support_bias": "medium",
        "contract_mix": {
            "electricity_only": "low",
            "gas_only": "low",
            "dual_fuel": "very_high",
        },
    },
    "healthcare": {
        "energy_intensity": "medium",
        "electricity_weight": "high",
        "gas_weight": "medium",
        "seasonality": {"electricity": "flat", "gas": "winter_heavy"},
        "revenue_volatility": "low",
        "digitalization_bias": "high",
        "support_bias": "medium",
        "contract_mix": {
            "electricity_only": "medium",
            "gas_only": "low",
            "dual_fuel": "high",
        },
    },
    "agriculture": {
        "energy_intensity": "low",
        "electricity_weight": "medium",
        "gas_weight": "medium",
        "seasonality": {"electricity": "summer_peak", "gas": "flat"},
        "revenue_volatility": "high",
        "digitalization_bias": "very_low",
        "support_bias": "low",
        "contract_mix": {
            "electricity_only": "high",
            "gas_only": "very_low",
            "dual_fuel": "low",
        },
    },
    "construction": {
        "energy_intensity": "low",
        "electricity_weight": "high",
        "gas_weight": "very_low",
        "seasonality": {"electricity": "summer_peak", "gas": "winter_heavy"},
        "revenue_volatility": "very_high",
        "digitalization_bias": "low",
        "support_bias": "medium",
        "contract_mix": {
            "electricity_only": "very_high",
            "gas_only": "very_low",
            "dual_fuel": "very_low",
        },
    },
    "transportation": {
        "energy_intensity": "medium",
        "electricity_weight": "high",
        "gas_weight": "low",
        "seasonality": {"electricity": "flat", "gas": "winter_heavy"},
        "revenue_volatility": "medium",
        "digitalization_bias": "high",
        "support_bias": "medium",
        "contract_mix": {
            "electricity_only": "high",
            "gas_only": "low",
            "dual_fuel": "medium",
        },
    },
    "utilities": {
        "energy_intensity": "high",
        "electricity_weight": "very_high",
        "gas_weight": "medium",
        "seasonality": {"electricity": "flat", "gas": "winter_heavy"},
        "revenue_volatility": "very_low",
        "digitalization_bias": "high",
        "support_bias": "very_high",
        "contract_mix": {
            "electricity_only": "high",
            "gas_only": "very_low",
            "dual_fuel": "low",
        },
    },
    "tech": {
        "energy_intensity": "medium",
        "electricity_weight": "very_high",
        "gas_weight": "very_low",
        "seasonality": {"electricity": "flat", "gas": "flat"},
        "revenue_volatility": "medium",
        "digitalization_bias": "very_high",
        "support_bias": "low",
        "contract_mix": {
            "electricity_only": "very_high",
            "gas_only": "very_low",
            "dual_fuel": "very_low",
        },
    },
}
