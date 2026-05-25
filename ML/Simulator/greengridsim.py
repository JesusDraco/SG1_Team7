import simpy
import math
import random
import json
import copy
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from ml_solar_model import SolarMLPredictor

print("\nLoading ML Solar Predictor...")
ML_SOLAR_PREDICTOR = SolarMLPredictor()
print("ML Solar Predictor loaded successfully.")

class Config:
    def __init__(self, data=None):
        data = data or {}

        simulation = data.get("simulation", {})
        grid = data.get("grid", {})
        battery = data.get("battery", {})
        solar = data.get("solar", {})
        neighborhood = data.get("neighborhood", {})

        self.TIME_STEP_MIN = simulation.get("time_step_min", 60)
        self.SIM_DAYS = simulation.get("sim_days", 30)
        self.SEASON = simulation.get("season", "Spring")
        self.STRATEGY = simulation.get("strategy", "LOAD_PRIORITY")
        self.RANDOM_SEED = simulation.get("random_seed", 42)

        self.BATTERY_CAPACITY = battery.get("capacity_kwh", 13.5)
        self.BATTERY_EFFICIENCY = battery.get("efficiency", 0.9)
        self.SOLAR_PEAK_KW = solar.get("solar_peak_kw", 5)
        self.BASE_LOAD_KW = 0.5

        self.ENFORCE_SOC_FLOOR = battery.get("enforce_soc_floor", False)
        self.SOC_FLOOR_FRAC = battery.get("soc_floor_frac", 0.05)

        self.INVERTER_MAX_OUTPUT_KW = solar.get("inverter_max_output_kw", 4)
        self.INVERTER_MTTF_DAYS = solar.get("inverter_mttf_days", 200)
        self.INVERTER_DOWNTIME_MIN_H = solar.get("inverter_downtime_min_h", 4)
        self.INVERTER_DOWNTIME_MAX_H = solar.get("inverter_downtime_max_h", 72)

        self.GRID_EXPORT_LIMIT_KW = grid.get("grid_export_limit_kw", 20)
        self.GRID_IMPORT_LIMIT_KW = grid.get("grid_import_limit_kw", 20)
        self.ZERO_EXPORT = grid.get("zero_export", False)

        self.IMPORT_COST = grid.get("import_cost", 0.75)
        self.EXPORT_REVENUE = grid.get("export_revenue", 0.9)

        self.NUM_HOUSEHOLDS = neighborhood.get("num_households", 1)

    def clone(self):
        return copy.deepcopy(self)


def dt_hours(config) -> float:
    return config.TIME_STEP_MIN / 60.0


def load_config(config_path="config.json"):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def ensure_output_dirs():
    output_dir = Path("output")
    dashboard_dir = output_dir / "dashboard_data"

    output_dir.mkdir(exist_ok=True)
    dashboard_dir.mkdir(exist_ok=True)

    return output_dir, dashboard_dir


def build_household_definitions(config_data):
    household_types = config_data.get("household_types", [])
    wealth_levels = config_data.get("wealth_levels", {})
    num_households = config_data.get("neighborhood", {}).get("num_households", len(household_types))

    if not household_types:
        raise ValueError("El config.json no contiene 'household_types'")

    households = []

    for i in range(num_households):
        profile = household_types[i % len(household_types)]

        wealth_name = profile.get("wealth_level", "middle")
        wealth_multiplier = wealth_levels.get(wealth_name, 1.0)

        households.append({
            "household_id": i + 1,
            "household_type": profile.get("type", f"type_{i+1}"),
            "wealth_level": wealth_name,
            "wealth_multiplier": wealth_multiplier,
            "base_load_kw": profile.get("base_load_kw", 0.5) * wealth_multiplier,
            "solar_peak_kw": profile.get("solar_peak_kw", 5.0),
            "battery_capacity_kwh": profile.get("battery_capacity_kwh", 13.5)
        })

    return households


def build_household_config(base_config, household_definition):
    cfg = base_config.clone()

    cfg.BASE_LOAD_KW = household_definition["base_load_kw"]
    cfg.SOLAR_PEAK_KW = household_definition["solar_peak_kw"]
    cfg.BATTERY_CAPACITY = household_definition["battery_capacity_kwh"]

    return cfg


class Weather:
    def __init__(self, env, config, season="Summer"):
        self.env = env
        self.config = config
        self.season = season
        self.cloud_coverage = 0.0
        self.process = env.process(self.run())

        self.season_weights = {
            "Spring": (0.1, 0.3, 0.4, 0.2),
            "Summer": (0.05, 0.15, 0.3, 0.5),
            "Fall":   (0.2, 0.4, 0.3, 0.1),
            "Winter": (0.3, 0.4, 0.2, 0.1)
        }

    def choose_daily_cloud(self):
        weights = self.season_weights[self.season]
        sky_type = random.choices(
            ["Clear", "Partly", "Mostly", "Overcast"],
            weights=weights
        )[0]

        if sky_type == "Clear":
            return random.uniform(0.0, 0.2)
        elif sky_type == "Partly":
            return random.uniform(0.2, 0.6)
        elif sky_type == "Mostly":
            return random.uniform(0.6, 0.8)
        else:
            return random.uniform(0.8, 0.9)

    def run(self):
        while True:
            self.cloud_coverage = self.choose_daily_cloud()
            yield self.env.timeout(24)


class Inverter:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        self.max_output = config.INVERTER_MAX_OUTPUT_KW
        self.failed = False
        self.total_failures = 0
        self.total_downtime = 0.0
        self.process = env.process(self.run())

    def run(self):
        while True:
            mean_hours = self.config.INVERTER_MTTF_DAYS * 24
            time_to_failure = random.expovariate(1 / mean_hours)
            yield self.env.timeout(time_to_failure)

            self.failed = True
            self.total_failures += 1

            downtime = random.uniform(
                self.config.INVERTER_DOWNTIME_MIN_H,
                self.config.INVERTER_DOWNTIME_MAX_H
            )
            self.total_downtime += downtime
            yield self.env.timeout(downtime)

            self.failed = False


class Solar:
    def __init__(self, env, config, weather, inverter):
        self.env = env
        self.config = config
        self.weather = weather
        self.inverter = inverter

        self.current_generation = 0.0  # kW

        self.ml_predictor = ML_SOLAR_PREDICTOR

        self.process = env.process(self.run())

    def run(self):
        while True:
            sim_hour = self.env.now

            predicted_generation_kw = self.ml_predictor.predict_generation_kw(
                sim_time_hours=sim_hour,
                household_peak_kw=self.config.SOLAR_PEAK_KW
            )

            if self.inverter.failed:
                predicted_generation_kw = 0.0

            self.current_generation = min(
                predicted_generation_kw,
                self.inverter.max_output
            )

            yield self.env.timeout(dt_hours(self.config))


class Load:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.current_load = 0.0  # kW
        self.process = env.process(self.run())

    def run(self):
        while True:
            hour = self.env.now % 24
            load = self.config.BASE_LOAD_KW

            if 18 <= hour <= 21:
                load += random.uniform(0, 3)

            self.current_load = load
            yield self.env.timeout(dt_hours(self.config))


class Battery:
    def __init__(self, config):
        self.capacity = config.BATTERY_CAPACITY
        self.roundtrip_eff = config.BATTERY_EFFICIENCY
        self.eta_c = math.sqrt(self.roundtrip_eff)
        self.eta_d = math.sqrt(self.roundtrip_eff)

        self.soc = 0.5 * self.capacity  # kWh

        self.enforce_floor = config.ENFORCE_SOC_FLOOR
        self.soc_floor = config.SOC_FLOOR_FRAC * self.capacity

        self.energy_in_kwh = 0.0
        self.energy_out_kwh = 0.0

    def charge(self, input_energy_kwh):
        if input_energy_kwh <= 0:
            return 0.0

        available_space = self.capacity - self.soc
        max_input_accepted = available_space / self.eta_c
        input_used = min(input_energy_kwh, max_input_accepted)

        stored = input_used * self.eta_c
        self.soc += stored

        self.energy_in_kwh += input_used
        return input_used

    def discharge(self, demand_energy_kwh):
        if demand_energy_kwh <= 0:
            return 0.0

        available_soc = self.soc
        if self.enforce_floor:
            available_soc = max(0.0, self.soc - self.soc_floor)

        max_deliverable = available_soc * self.eta_d
        delivered = min(demand_energy_kwh, max_deliverable)

        soc_used = delivered / self.eta_d
        self.soc -= soc_used

        self.energy_out_kwh += delivered
        return delivered


class Grid:
    def __init__(self, config):
        self.config = config
        self.total_import = 0.0
        self.total_export = 0.0

    def import_energy(self, energy_kwh):
        step_kwh_limit = self.config.GRID_IMPORT_LIMIT_KW * dt_hours(self.config)
        imported = min(energy_kwh, step_kwh_limit)
        self.total_import += imported
        return imported

    def export_energy(self, energy_kwh):
        if self.config.ZERO_EXPORT:
            return 0.0

        step_kwh_limit = self.config.GRID_EXPORT_LIMIT_KW * dt_hours(self.config)
        exported = min(energy_kwh, step_kwh_limit)
        self.total_export += exported
        return exported


class HomeSystem:
    def __init__(self, env, config, season="Summer", household_info=None):
        self.env = env
        self.config = config
        self.household_info = household_info or {}

        self.weather = Weather(env, config, season)
        self.inverter = Inverter(env, config)
        self.solar = Solar(env, config, self.weather, self.inverter)
        self.load = Load(env, config)
        self.battery = Battery(config)
        self.grid = Grid(config)

        self.log = []
        self.process = env.process(self.run())

    def energy_flow(self, generation_kwh, demand_kwh):
        grid_import = 0.0
        grid_export = 0.0
        curtailed_kwh = 0.0
        unmet_load_kwh = 0.0

        solar_used_kwh = 0.0
        battery_charge_kwh = 0.0
        battery_discharge_kwh = 0.0

        strategy = self.config.STRATEGY

        if strategy == "LOAD_PRIORITY":
            solar_used_kwh = min(generation_kwh, demand_kwh)
            remaining_demand = demand_kwh - solar_used_kwh
            excess_solar = generation_kwh - solar_used_kwh

            if remaining_demand > 0:
                battery_discharge_kwh = self.battery.discharge(remaining_demand)
                remaining_demand -= battery_discharge_kwh

                if remaining_demand > 0:
                    imported = self.grid.import_energy(remaining_demand)
                    grid_import = imported
                    remaining_demand -= imported

                    if remaining_demand > 0:
                        unmet_load_kwh += remaining_demand

            if excess_solar > 0:
                battery_charge_kwh = self.battery.charge(excess_solar)
                excess_solar -= battery_charge_kwh

                if excess_solar > 0:
                    exported = self.grid.export_energy(excess_solar)
                    grid_export = exported
                    curtailed_kwh += max(0.0, excess_solar - exported)

        elif strategy == "CHARGE_PRIORITY":
            battery_charge_kwh = self.battery.charge(generation_kwh)
            remaining_solar = generation_kwh - battery_charge_kwh

            solar_used_kwh = min(remaining_solar, demand_kwh)
            remaining_demand = demand_kwh - solar_used_kwh

            if remaining_demand > 0:
                battery_discharge_kwh = self.battery.discharge(remaining_demand)
                remaining_demand -= battery_discharge_kwh

                if remaining_demand > 0:
                    imported = self.grid.import_energy(remaining_demand)
                    grid_import = imported
                    remaining_demand -= imported

                    if remaining_demand > 0:
                        unmet_load_kwh += remaining_demand

            excess_solar = remaining_solar - solar_used_kwh
            if excess_solar > 0:
                exported = self.grid.export_energy(excess_solar)
                grid_export = exported
                curtailed_kwh += max(0.0, excess_solar - exported)

        elif strategy == "PRODUCE_PRIORITY":
            exported = self.grid.export_energy(generation_kwh)
            grid_export = exported
            curtailed_kwh += max(0.0, generation_kwh - exported)

            remaining_demand = demand_kwh

            battery_discharge_kwh = self.battery.discharge(remaining_demand)
            remaining_demand -= battery_discharge_kwh

            if remaining_demand > 0:
                imported = self.grid.import_energy(remaining_demand)
                grid_import = imported
                remaining_demand -= imported

                if remaining_demand > 0:
                    unmet_load_kwh += remaining_demand

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self_consumed_kwh = solar_used_kwh + battery_discharge_kwh
        net_load_kwh = demand_kwh - generation_kwh

        return {
            "grid_import": grid_import,
            "grid_export": grid_export,
            "curtailed_kwh": curtailed_kwh,
            "unmet_load_kwh": unmet_load_kwh,
            "solar_used_kwh": solar_used_kwh,
            "battery_charge_kwh": battery_charge_kwh,
            "battery_discharge_kwh": battery_discharge_kwh,
            "self_consumed_kwh": self_consumed_kwh,
            "net_load_kwh": net_load_kwh
        }

    def run(self):
        while True:
            dt = dt_hours(self.config)

            generation_kw = self.solar.current_generation
            demand_kw = self.load.current_load

            generation_kwh = generation_kw * dt
            demand_kwh = demand_kw * dt

            flow = self.energy_flow(generation_kwh, demand_kwh)

            self.log.append({
                "household_id": self.household_info.get("household_id"),
                "household_type": self.household_info.get("household_type"),
                "wealth_level": self.household_info.get("wealth_level"),
                "wealth_multiplier": self.household_info.get("wealth_multiplier"),

                "time": self.env.now,

                "generation_kw": generation_kw,
                "load_kw": demand_kw,

                "generation_kwh": generation_kwh,
                "load_kwh": demand_kwh,
                "solar_used_kwh": flow["solar_used_kwh"],
                "battery_charge_kwh": flow["battery_charge_kwh"],
                "battery_discharge_kwh": flow["battery_discharge_kwh"],
                "self_consumed_kwh": flow["self_consumed_kwh"],
                "net_load_kwh": flow["net_load_kwh"],

                "soc": self.battery.soc,
                "cloud": self.weather.cloud_coverage,
                "inverter_failed": self.inverter.failed,

                "ml_generation_kw": generation_kw,

                "grid_import": flow["grid_import"],
                "grid_export": flow["grid_export"],
                "curtailed_kwh": flow["curtailed_kwh"],
                "unmet_load_kwh": flow["unmet_load_kwh"],

                "strategy": self.config.STRATEGY
            })

            yield self.env.timeout(dt)


def compute_metrics(df, home, config, season, strategy):
    total_generation = df["generation_kwh"].sum()
    total_consumption = df["load_kwh"].sum()

    total_import = df["grid_import"].sum()
    total_export = df["grid_export"].sum()

    total_unmet = df["unmet_load_kwh"].sum()
    total_curtail = df["curtailed_kwh"].sum()

    total_solar_used = df["solar_used_kwh"].sum()
    total_battery_charge = df["battery_charge_kwh"].sum()
    total_battery_discharge = df["battery_discharge_kwh"].sum()
    total_self_consumed = df["self_consumed_kwh"].sum()

    avg_net_load = df["net_load_kwh"].mean()
    max_net_load = df["net_load_kwh"].max()
    min_net_load = df["net_load_kwh"].min()

    avg_soc = df["soc"].mean()
    min_soc = df["soc"].min()
    max_soc = df["soc"].max()

    avg_cloud = df["cloud"].mean()

    peak_load_kw = df["load_kw"].max()
    peak_gen_kw = df["generation_kw"].max()

    soc_full_events = (df["soc"] >= (config.BATTERY_CAPACITY - 1e-6)).sum()
    soc_empty_events = (df["soc"] <= (0.0 + 1e-6)).sum()

    unmet_events = (df["unmet_load_kwh"] > 0).sum()
    total_steps = len(df)
    unmet_pct_time = (unmet_events / total_steps) if total_steps > 0 else 0.0
    unmet_hours = unmet_events * dt_hours(config)

    avg_generation_kwh_per_day = total_generation / config.SIM_DAYS
    avg_consumption_kwh_per_day = total_consumption / config.SIM_DAYS
    avg_generation_kwh_per_step = df["generation_kwh"].mean()
    avg_consumption_kwh_per_step = df["load_kwh"].mean()

    inverter_failed_hours_step = df["inverter_failed"].astype(int).sum() * dt_hours(config)

    inverter_failures = home.inverter.total_failures
    inverter_downtime_hours = home.inverter.total_downtime
    avg_failure_duration = (inverter_downtime_hours / inverter_failures) if inverter_failures > 0 else 0.0

    batt_in = home.battery.energy_in_kwh
    batt_out = home.battery.energy_out_kwh
    batt_eff_real = (batt_out / batt_in) if batt_in > 0 else 0.0

    net_cost = total_import * config.IMPORT_COST - total_export * config.EXPORT_REVENUE

    self_consumption_ratio = (total_self_consumed / total_generation) if total_generation > 0 else 0.0
    self_sufficiency_ratio = (total_self_consumed / total_consumption) if total_consumption > 0 else 0.0

    return {
        "season": season,
        "strategy": strategy,

        "generation_kwh": total_generation,
        "consumption_kwh": total_consumption,

        "avg_generation_kwh_per_day": avg_generation_kwh_per_day,
        "avg_consumption_kwh_per_day": avg_consumption_kwh_per_day,
        "avg_generation_kwh_per_step": avg_generation_kwh_per_step,
        "avg_consumption_kwh_per_step": avg_consumption_kwh_per_step,

        "grid_import_kwh": total_import,
        "grid_export_kwh": total_export,

        "curtailed_kwh": total_curtail,
        "unmet_load_kwh": total_unmet,

        "solar_used_kwh": total_solar_used,
        "battery_charge_kwh": total_battery_charge,
        "battery_discharge_kwh": total_battery_discharge,
        "self_consumed_kwh": total_self_consumed,

        "self_consumption_ratio": self_consumption_ratio,
        "self_sufficiency_ratio": self_sufficiency_ratio,

        "avg_net_load_kwh": avg_net_load,
        "max_net_load_kwh": max_net_load,
        "min_net_load_kwh": min_net_load,

        "unmet_events": int(unmet_events),
        "unmet_pct_time": unmet_pct_time,
        "unmet_hours": unmet_hours,

        "avg_soc_kwh": avg_soc,
        "min_soc_kwh": min_soc,
        "max_soc_kwh": max_soc,

        "soc_full_events": int(soc_full_events),
        "soc_empty_events": int(soc_empty_events),

        "avg_cloud": avg_cloud,
        "peak_load_kw": peak_load_kw,
        "peak_gen_kw": peak_gen_kw,

        "inverter_failed_hours_step": inverter_failed_hours_step,
        "inverter_failures": inverter_failures,
        "inverter_downtime_hours": inverter_downtime_hours,
        "avg_failure_duration_hours": avg_failure_duration,

        "batt_in_kwh": batt_in,
        "batt_out_kwh": batt_out,
        "batt_eff_real": batt_eff_real,

        "net_cost": net_cost
    }


def simulate_neighborhood(base_config, household_definitions, season, strategy):
    all_logs = []
    household_metrics = []

    for household in household_definitions:
        env = simpy.Environment()

        house_config = build_household_config(base_config, household)
        house_config.STRATEGY = strategy

        home = HomeSystem(
            env,
            house_config,
            season=season,
            household_info=household
        )

        env.run(until=24 * house_config.SIM_DAYS)

        df_house = pd.DataFrame(home.log)
        all_logs.append(df_house)

        metrics = compute_metrics(df_house, home, house_config, season, strategy)
        metrics["household_id"] = household["household_id"]
        metrics["household_type"] = household["household_type"]
        metrics["wealth_level"] = household["wealth_level"]
        metrics["wealth_multiplier"] = household["wealth_multiplier"]
        household_metrics.append(metrics)

    df_all = pd.concat(all_logs, ignore_index=True)
    metrics_df = pd.DataFrame(household_metrics)

    return df_all, metrics_df


def aggregate_neighborhood_hourly(df_all):
    grouped = df_all.groupby("time", as_index=False).agg({
        "generation_kw": "sum",
        "load_kw": "sum",
        "generation_kwh": "sum",
        "load_kwh": "sum",
        "solar_used_kwh": "sum",
        "battery_charge_kwh": "sum",
        "battery_discharge_kwh": "sum",
        "self_consumed_kwh": "sum",
        "net_load_kwh": "sum",
        "grid_import": "sum",
        "grid_export": "sum",
        "curtailed_kwh": "sum",
        "unmet_load_kwh": "sum",
        "soc": "sum"
    })

    grouped = grouped.rename(columns={
        "soc": "total_soc_kwh"
    })

    return grouped


def build_dashboard_kpis(df_all, household_metrics_df, season, strategy):
    total_generation = df_all["generation_kwh"].sum()
    total_consumption = df_all["load_kwh"].sum()
    total_import = df_all["grid_import"].sum()
    total_export = df_all["grid_export"].sum()
    total_self_consumed = df_all["self_consumed_kwh"].sum()
    total_unmet = df_all["unmet_load_kwh"].sum()
    total_curtailed = df_all["curtailed_kwh"].sum()

    avg_self_consumption_ratio = household_metrics_df["self_consumption_ratio"].mean()
    avg_self_sufficiency_ratio = household_metrics_df["self_sufficiency_ratio"].mean()
    avg_net_cost_per_house = household_metrics_df["net_cost"].mean()
    total_net_cost = household_metrics_df["net_cost"].sum()

    kpis = pd.DataFrame([{
        "season": season,
        "strategy": strategy,
        "num_households": int(df_all["household_id"].nunique()),
        "total_generation_kwh": total_generation,
        "total_consumption_kwh": total_consumption,
        "total_grid_import_kwh": total_import,
        "total_grid_export_kwh": total_export,
        "total_self_consumed_kwh": total_self_consumed,
        "total_unmet_load_kwh": total_unmet,
        "total_curtailed_kwh": total_curtailed,
        "avg_self_consumption_ratio": avg_self_consumption_ratio,
        "avg_self_sufficiency_ratio": avg_self_sufficiency_ratio,
        "avg_net_cost_per_house": avg_net_cost_per_house,
        "total_net_cost": total_net_cost
    }])

    return kpis


def build_by_household_type(df_all, household_metrics_df, season, strategy):
    summary = household_metrics_df.groupby("household_type", as_index=False).agg({
        "generation_kwh": "mean",
        "consumption_kwh": "mean",
        "grid_import_kwh": "mean",
        "grid_export_kwh": "mean",
        "self_consumed_kwh": "mean",
        "battery_charge_kwh": "mean",
        "battery_discharge_kwh": "mean",
        "net_cost": "mean",
        "self_consumption_ratio": "mean",
        "self_sufficiency_ratio": "mean",
        "avg_soc_kwh": "mean"
    })

    summary["season"] = season
    summary["strategy"] = strategy

    return summary


def build_by_wealth_level(df_all, household_metrics_df, season, strategy):
    summary = household_metrics_df.groupby("wealth_level", as_index=False).agg({
        "generation_kwh": "mean",
        "consumption_kwh": "mean",
        "grid_import_kwh": "mean",
        "grid_export_kwh": "mean",
        "self_consumed_kwh": "mean",
        "battery_charge_kwh": "mean",
        "battery_discharge_kwh": "mean",
        "net_cost": "mean",
        "self_consumption_ratio": "mean",
        "self_sufficiency_ratio": "mean",
        "avg_soc_kwh": "mean"
    })

    summary["season"] = season
    summary["strategy"] = strategy

    return summary


def build_duck_curve_dataset(df_hourly, season, strategy):
    duck_df = df_hourly.copy()
    duck_df["season"] = season
    duck_df["strategy"] = strategy

    return duck_df[[
        "time",
        "generation_kwh",
        "load_kwh",
        "self_consumed_kwh",
        "grid_import",
        "grid_export",
        "net_load_kwh",
        "season",
        "strategy"
    ]]


def build_costs_and_self_consumption_dataset(household_metrics_df, season, strategy):
    df = household_metrics_df.copy()
    df["season"] = season
    df["strategy"] = strategy

    return df[[
        "household_id",
        "household_type",
        "wealth_level",
        "generation_kwh",
        "consumption_kwh",
        "self_consumed_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "net_cost",
        "self_consumption_ratio",
        "self_sufficiency_ratio",
        "season",
        "strategy"
    ]]


def build_timeseries_by_household_type(df_all, season, strategy):
    grouped = df_all.groupby(["time", "household_type"], as_index=False).agg({
        "generation_kwh": "sum",
        "load_kwh": "sum",
        "self_consumed_kwh": "sum",
        "grid_import": "sum",
        "grid_export": "sum",
        "net_load_kwh": "sum"
    })

    grouped["season"] = season
    grouped["strategy"] = strategy

    return grouped


def export_neighborhood_csv(df_all, df_hourly, household_metrics_df, season, strategy, config):
    output_dir, dashboard_dir = ensure_output_dirs()

    detailed_name = output_dir / f"greengridsim_neighborhood_detailed_{season}_{strategy}_step{config.TIME_STEP_MIN}min.csv"
    hourly_name = output_dir / f"greengridsim_neighborhood_hourly_{season}_{strategy}_step{config.TIME_STEP_MIN}min.csv"
    metrics_name = output_dir / f"greengridsim_household_metrics_{season}_{strategy}_step{config.TIME_STEP_MIN}min.csv"

    df_all.to_csv(detailed_name, index=False)
    df_hourly.to_csv(hourly_name, index=False)
    household_metrics_df.to_csv(metrics_name, index=False)

    print(f"\nCSV generado: {detailed_name}")
    print(f"CSV generado: {hourly_name}")
    print(f"CSV generado: {metrics_name}")

    kpis_df = build_dashboard_kpis(df_all, household_metrics_df, season, strategy)
    by_household_type_df = build_by_household_type(df_all, household_metrics_df, season, strategy)
    by_wealth_level_df = build_by_wealth_level(df_all, household_metrics_df, season, strategy)
    duck_curve_df = build_duck_curve_dataset(df_hourly, season, strategy)
    costs_self_df = build_costs_and_self_consumption_dataset(household_metrics_df, season, strategy)
    timeseries_type_df = build_timeseries_by_household_type(df_all, season, strategy)

    kpis_df.to_csv(dashboard_dir / f"kpis_{season}_{strategy}.csv", index=False)
    by_household_type_df.to_csv(dashboard_dir / f"by_household_type_{season}_{strategy}.csv", index=False)
    by_wealth_level_df.to_csv(dashboard_dir / f"by_wealth_level_{season}_{strategy}.csv", index=False)
    duck_curve_df.to_csv(dashboard_dir / f"duck_curve_{season}_{strategy}.csv", index=False)
    costs_self_df.to_csv(dashboard_dir / f"costs_self_consumption_{season}_{strategy}.csv", index=False)
    timeseries_type_df.to_csv(dashboard_dir / f"timeseries_by_household_type_{season}_{strategy}.csv", index=False)

    kpis_df.to_json(dashboard_dir / f"kpis_{season}_{strategy}.json", orient="records", indent=2)
    by_household_type_df.to_json(dashboard_dir / f"by_household_type_{season}_{strategy}.json", orient="records", indent=2)
    by_wealth_level_df.to_json(dashboard_dir / f"by_wealth_level_{season}_{strategy}.json", orient="records", indent=2)
    duck_curve_df.to_json(dashboard_dir / f"duck_curve_{season}_{strategy}.json", orient="records", indent=2)
    costs_self_df.to_json(dashboard_dir / f"costs_self_consumption_{season}_{strategy}.json", orient="records", indent=2)
    timeseries_type_df.to_json(dashboard_dir / f"timeseries_by_household_type_{season}_{strategy}.json", orient="records", indent=2)

    print(f"\nDatasets de dashboard generados en: {dashboard_dir}")

config_data = load_config("config.json")
base_config = Config(config_data)

random.seed(base_config.RANDOM_SEED)

household_definitions = build_household_definitions(config_data)

strategies = ["LOAD_PRIORITY", "CHARGE_PRIORITY", "PRODUCE_PRIORITY"]
seasons = ["Spring", "Summer", "Fall", "Winter"]

all_strategy_results = []
all_household_metrics = []

for season in seasons:
    for strategy in strategies:
        print(f"\n===== Running neighborhood simulation | Season: {season} | Strategy: {strategy} =====")

        df_all, household_metrics_df = simulate_neighborhood(
            base_config=base_config,
            household_definitions=household_definitions,
            season=season,
            strategy=strategy
        )

        df_hourly = aggregate_neighborhood_hourly(df_all)

        export_neighborhood_csv(
            df_all=df_all,
            df_hourly=df_hourly,
            household_metrics_df=household_metrics_df,
            season=season,
            strategy=strategy,
            config=base_config
        )

        neighborhood_summary = {
            "season": season,
            "strategy": strategy,
            "num_households": len(household_definitions),

            "total_generation_kwh": df_all["generation_kwh"].sum(),
            "total_consumption_kwh": df_all["load_kwh"].sum(),
            "total_solar_used_kwh": df_all["solar_used_kwh"].sum(),
            "total_battery_charge_kwh": df_all["battery_charge_kwh"].sum(),
            "total_battery_discharge_kwh": df_all["battery_discharge_kwh"].sum(),
            "total_self_consumed_kwh": df_all["self_consumed_kwh"].sum(),

            "total_grid_import_kwh": df_all["grid_import"].sum(),
            "total_grid_export_kwh": df_all["grid_export"].sum(),
            "total_curtailed_kwh": df_all["curtailed_kwh"].sum(),
            "total_unmet_load_kwh": df_all["unmet_load_kwh"].sum(),

            "avg_net_load_kwh": df_all["net_load_kwh"].mean(),
            "max_net_load_kwh": df_all["net_load_kwh"].max(),
            "min_net_load_kwh": df_all["net_load_kwh"].min(),

            "avg_soc_kwh_per_house": household_metrics_df["avg_soc_kwh"].mean(),
            "avg_net_cost_per_house": household_metrics_df["net_cost"].mean(),
            "total_net_cost": household_metrics_df["net_cost"].sum(),

            "avg_self_consumption_ratio": household_metrics_df["self_consumption_ratio"].mean(),
            "avg_self_sufficiency_ratio": household_metrics_df["self_sufficiency_ratio"].mean()
        }

        all_strategy_results.append(neighborhood_summary)
        all_household_metrics.append(household_metrics_df)

        plt.figure(figsize=(12, 4))
        plt.plot(df_hourly["time"], df_hourly["load_kwh"], label="Neighborhood Load (kWh)")
        plt.plot(df_hourly["time"], df_hourly["generation_kwh"], label="Neighborhood Solar (kWh)")
        plt.plot(df_hourly["time"], df_hourly["net_load_kwh"], label="Net Load (kWh)")
        plt.title(f"Duck Curve - {strategy} - {season}")
        plt.xlabel("Time (hours)")
        plt.ylabel("Energy (kWh)")
        plt.grid()
        plt.legend()
        #plt.show()

        plt.figure(figsize=(12, 4))
        plt.plot(df_hourly["time"], df_hourly["grid_import"], label="Grid Import (kWh)")
        plt.plot(df_hourly["time"], df_hourly["grid_export"], label="Grid Export (kWh)")
        plt.title(f"Neighborhood Grid Import / Export - {strategy} - {season}")
        plt.xlabel("Time (hours)")
        plt.ylabel("Energy (kWh)")
        plt.grid()
        plt.legend()
        #plt.show()

        consumption_by_type = df_all.groupby("household_type", as_index=False)["load_kwh"].sum()

        plt.figure(figsize=(10, 4))
        plt.bar(consumption_by_type["household_type"], consumption_by_type["load_kwh"])
        plt.title(f"Consumption by Household Type - {strategy} - {season}")
        plt.xlabel("Household Type")
        plt.ylabel("Total Consumption (kWh)")
        plt.grid(axis="y")
        #plt.show()

        export_by_wealth = df_all.groupby("wealth_level", as_index=False)["grid_export"].sum()

        plt.figure(figsize=(10, 4))
        plt.bar(export_by_wealth["wealth_level"], export_by_wealth["grid_export"])
        plt.title(f"Grid Export by Wealth Level - {strategy} - {season}")
        plt.xlabel("Wealth Level")
        plt.ylabel("Total Export (kWh)")
        plt.grid(axis="y")
        #plt.show()

results_df = pd.DataFrame(all_strategy_results)
all_household_metrics_df = pd.concat(all_household_metrics, ignore_index=True)

print("\n===== NEIGHBORHOOD RESULTS SUMMARY =====")
print(results_df)

print("\n===== HOUSEHOLD METRICS SAMPLE =====")
print(all_household_metrics_df.head())

output_dir, dashboard_dir = ensure_output_dirs()

results_summary_path = output_dir / "greengridsim_neighborhood_results_summary.csv"
household_metrics_path = output_dir / "greengridsim_all_household_metrics.csv"

results_df.to_csv(results_summary_path, index=False)
all_household_metrics_df.to_csv(household_metrics_path, index=False)

results_df.to_json(dashboard_dir / "all_results_summary.json", orient="records", indent=2)
all_household_metrics_df.to_json(dashboard_dir / "all_household_metrics.json", orient="records", indent=2)

print(f"\nCSV generado: {results_summary_path}")
print(f"CSV generado: {household_metrics_path}")
print(f"JSON generado: {dashboard_dir / 'all_results_summary.json'}")
print(f"JSON generado: {dashboard_dir / 'all_household_metrics.json'}")