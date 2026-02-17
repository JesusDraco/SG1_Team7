import simpy
import math
import random
import pandas as pd
import matplotlib.pyplot as plt

class Config:
    # Tiempo y horizonte
    TIME_STEP_MIN = 60      
    SIM_DAYS = 30

    # Componentes fisicos
    BATTERY_CAPACITY = 13.5
    BATTERY_EFFICIENCY = 0.9   
    SOLAR_PEAK_KW = 5
    BASE_LOAD_KW = 0.5

    # Piso de bateria
    ENFORCE_SOC_FLOOR = False
    SOC_FLOOR_FRAC = 0.05   

    # Inversor
    INVERTER_MAX_OUTPUT_KW = 4
    INVERTER_MTTF_DAYS = 200
    INVERTER_DOWNTIME_MIN_H = 4
    INVERTER_DOWNTIME_MAX_H = 72

    # Red
    GRID_EXPORT_LIMIT_KW = 20
    GRID_IMPORT_LIMIT_KW = 20
    ZERO_EXPORT = False

    IMPORT_COST = 0.75
    EXPORT_REVENUE = 0.9

    STRATEGY = "LOAD_PRIORITY"


def dt_hours(config) -> float:
    return config.TIME_STEP_MIN / 60.0

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
        self.process = env.process(self.run())

    def run(self):
        while True:
            hour = self.env.now % 24
            sun_angle = hour * (math.pi / 12)

            ideal_generation = self.config.SOLAR_PEAK_KW * max(0.0, math.sin(sun_angle))
            adjusted_generation = ideal_generation * (1.0 - self.weather.cloud_coverage)

            if self.inverter.failed:
                adjusted_generation = 0.0

            self.current_generation = min(adjusted_generation, self.inverter.max_output)

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
    def __init__(self, env, config, season="Summer"):
        self.env = env
        self.config = config

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

        strategy = self.config.STRATEGY

        if strategy == "LOAD_PRIORITY":
            solar_used = min(generation_kwh, demand_kwh)
            remaining_demand = demand_kwh - solar_used
            excess_solar = generation_kwh - solar_used

            if remaining_demand > 0:
                battery_delivered = self.battery.discharge(remaining_demand)
                remaining_demand -= battery_delivered

                if remaining_demand > 0:
                    imported = self.grid.import_energy(remaining_demand)
                    grid_import = imported
                    remaining_demand -= imported

                    if remaining_demand > 0:
                        unmet_load_kwh += remaining_demand

            if excess_solar > 0:
                input_used = self.battery.charge(excess_solar)
                excess_solar -= input_used

                if excess_solar > 0:
                    exported = self.grid.export_energy(excess_solar)
                    grid_export = exported
                    curtailed_kwh += max(0.0, excess_solar - exported)

        elif strategy == "CHARGE_PRIORITY":
            input_used = self.battery.charge(generation_kwh)
            remaining_solar = generation_kwh - input_used

            solar_used = min(remaining_solar, demand_kwh)
            remaining_demand = demand_kwh - solar_used

            if remaining_demand > 0:
                battery_delivered = self.battery.discharge(remaining_demand)
                remaining_demand -= battery_delivered

                if remaining_demand > 0:
                    imported = self.grid.import_energy(remaining_demand)
                    grid_import = imported
                    remaining_demand -= imported
                    if remaining_demand > 0:
                        unmet_load_kwh += remaining_demand

            excess_solar = remaining_solar - solar_used
            if excess_solar > 0:
                exported = self.grid.export_energy(excess_solar)
                grid_export = exported
                curtailed_kwh += max(0.0, excess_solar - exported)

        elif strategy == "PRODUCE_PRIORITY":
            exported = self.grid.export_energy(generation_kwh)
            grid_export = exported
            curtailed_kwh += max(0.0, generation_kwh - exported)

            remaining_demand = demand_kwh

            battery_delivered = self.battery.discharge(remaining_demand)
            remaining_demand -= battery_delivered

            if remaining_demand > 0:
                imported = self.grid.import_energy(remaining_demand)
                grid_import = imported
                remaining_demand -= imported
                if remaining_demand > 0:
                    unmet_load_kwh += remaining_demand
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return grid_import, grid_export, curtailed_kwh, unmet_load_kwh

    def run(self):
        while True:
            dt = dt_hours(self.config)  # horas por paso

            generation_kw = self.solar.current_generation
            demand_kw = self.load.current_load

            generation_kwh = generation_kw * dt
            demand_kwh = demand_kw * dt

            grid_import, grid_export, curtailed_kwh, unmet_load_kwh = self.energy_flow(
                generation_kwh, demand_kwh
            )

            self.log.append({
                "time": self.env.now,

                # Potencias
                "generation_kw": generation_kw,
                "load_kw": demand_kw,

                # Energias por paso
                "generation_kwh": generation_kwh,
                "load_kwh": demand_kwh,

                # Estado
                "soc": self.battery.soc,
                "cloud": self.weather.cloud_coverage,
                "inverter_failed": self.inverter.failed,

                # Red / metricas
                "grid_import": grid_import,
                "grid_export": grid_export,
                "curtailed_kwh": curtailed_kwh,
                "unmet_load_kwh": unmet_load_kwh,

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

    avg_soc = df["soc"].mean()
    min_soc = df["soc"].min()
    max_soc = df["soc"].max()

    avg_cloud = df["cloud"].mean()

    peak_load_kw = df["load_kw"].max()
    peak_gen_kw = df["generation_kw"].max()

    soc_full_events = (df["soc"] >= (config.BATTERY_CAPACITY - 1e-6)).sum()
    soc_empty_events = (df["soc"] <= (0.0 + 1e-6)).sum()

    # HOW OFTEN UNMET LOAD
    unmet_events = (df["unmet_load_kwh"] > 0).sum()
    total_steps = len(df)
    unmet_pct_time = (unmet_events / total_steps) if total_steps > 0 else 0.0
    unmet_hours = unmet_events * dt_hours(config)

    # AVERAGE PRODUCTION/CONSUMPTION
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

    return {
        "season": season,
        "strategy": strategy,

        "generation_kwh": total_generation,
        "consumption_kwh": total_consumption,

        # Averages
        "avg_generation_kwh_per_day": avg_generation_kwh_per_day,
        "avg_consumption_kwh_per_day": avg_consumption_kwh_per_day,
        "avg_generation_kwh_per_step": avg_generation_kwh_per_step,
        "avg_consumption_kwh_per_step": avg_consumption_kwh_per_step,

        "grid_import_kwh": total_import,
        "grid_export_kwh": total_export,

        "curtailed_kwh": total_curtail,
        "unmet_load_kwh": total_unmet,

        # Unmet frequency
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

        # Inverter stats
        "inverter_failed_hours_step": inverter_failed_hours_step,
        "inverter_failures": inverter_failures,
        "inverter_downtime_hours": inverter_downtime_hours,
        "avg_failure_duration_hours": avg_failure_duration,

        # Battery stats
        "batt_in_kwh": batt_in,
        "batt_out_kwh": batt_out,
        "batt_eff_real": batt_eff_real,

        # Economic
        "net_cost": net_cost
    }


def compute_cloud_impact(df):
    cloud_bins = pd.cut(
        df["cloud"],
        bins=[0.0, 0.2, 0.6, 0.8, 0.9, 1.0],
        labels=["Clear", "Partly", "Mostly", "Overcast", "Extreme"],
        include_lowest=True
    )

    impact = df.groupby(cloud_bins, observed=False).agg({
        "generation_kwh": "mean",
        "grid_import": "mean",
        "grid_export": "mean",
        "unmet_load_kwh": "mean",
        "curtailed_kwh": "mean",
        "soc": "mean"
    }).reset_index()

    impact = impact.rename(columns={"cloud": "cloud_bin"})
    impact = impact.rename(columns={impact.columns[0]: "cloud_bin"})
    return impact


def export_hourly_csv(df, config, season, strategy):
    """
    Genera un CSV con resumen por hora (1 fila por hora)
    Funciona con cualquier TIME_STEP_MIN (15, 30, 60, etc.)
    """
    df_hourly = df.copy()
    df_hourly["hour_index"] = df_hourly["time"].astype(int)

    hourly_summary = df_hourly.groupby("hour_index", as_index=False).agg({
        # Promedios de potencia (kW)
        "generation_kw": "mean",
        "load_kw": "mean",

        # Energía por hora (kWh) sumada en esa hora
        "generation_kwh": "sum",
        "load_kwh": "sum",
        "grid_import": "sum",
        "grid_export": "sum",
        "curtailed_kwh": "sum",
        "unmet_load_kwh": "sum",

        # Estados promedio
        "soc": "mean",
        "cloud": "mean",

        # Conteo de steps en falla dentro de la hora
        "inverter_failed": "sum"
    })

    # Convertir conteo de "failed steps" a fracción de la hora 
    dt = dt_hours(config)
    steps_per_hour = int(round(1.0 / dt)) if dt > 0 else 1
    hourly_summary["inverter_failed_frac"] = hourly_summary["inverter_failed"] / max(1, steps_per_hour)
    hourly_summary.drop(columns=["inverter_failed"], inplace=True)

    # Metadata
    hourly_summary["season"] = season
    hourly_summary["strategy"] = strategy
    hourly_summary["time_step_min"] = config.TIME_STEP_MIN

    csv_name = f"greengridsim_hourly_{season}_{strategy}_step{config.TIME_STEP_MIN}min.csv"
    hourly_summary.to_csv(csv_name, index=False)
    print(f"\nCSV generado: {csv_name}")

# Experimentos
strategies = ["LOAD_PRIORITY", "CHARGE_PRIORITY", "PRODUCE_PRIORITY"]
#strategies = [Config.STRATEGY]

#seasons = ["Spring", "Summer", "Fall", "Winter"]
seasons = ["Spring"]
#seasons = ["Summer"]
#seasons = ["Fall"]
#seasons = ["Winter"]

all_results = []
all_cloud_impacts = []

for season in seasons:
    for strategy in strategies:
        Config.STRATEGY = strategy

        env = simpy.Environment()
        home = HomeSystem(env, Config, season=season)

        env.run(until=24 * Config.SIM_DAYS)

        df = pd.DataFrame(home.log)

        export_hourly_csv(df, Config, season, strategy)

        all_results.append(compute_metrics(df, home, Config, season, strategy))

        impact = compute_cloud_impact(df)
        impact["season"] = season
        impact["strategy"] = strategy
        all_cloud_impacts.append(impact)

        # Graficas

        # SoC
        plt.figure(figsize=(12, 4))
        plt.plot(df["time"], df["soc"])
        plt.title(f"SoC (kWh) - {strategy} - {season}")
        plt.xlabel("Time (hours)")
        plt.ylabel("Battery SoC (kWh)")
        plt.grid()
        plt.show()

        # Solar vs Load (kW)
        plt.figure(figsize=(12, 4))
        plt.plot(df["time"], df["generation_kw"], label="Solar (kW)")
        plt.plot(df["time"], df["load_kw"], label="Load (kW)")
        plt.title(f"Solar vs Load (kW) - {strategy} - {season}")
        plt.xlabel("Time (hours)")
        plt.ylabel("kW")
        plt.grid()
        plt.legend()
        plt.show()

        # Grid import/export (kWh por paso)
        plt.figure(figsize=(12, 4))
        plt.plot(df["time"], df["grid_import"], label="Grid Import (kWh/step)")
        plt.plot(df["time"], df["grid_export"], label="Grid Export (kWh/step)")
        plt.title(f"Grid Import/Export (kWh/step) - {strategy} - {season}")
        plt.xlabel("Time (hours)")
        plt.ylabel("kWh per step")
        plt.grid()
        plt.legend()
        plt.show()

        # Unmet load (kWh por paso)
        plt.figure(figsize=(12, 4))
        plt.plot(df["time"], df["unmet_load_kwh"])
        plt.title(f"Unmet Load (kWh/step) - {strategy} - {season}")
        plt.xlabel("Time (hours)")
        plt.ylabel("kWh per step")
        plt.grid()
        plt.show()

        # Cloud coverage
        plt.figure(figsize=(12, 3))
        plt.plot(df["time"], df["cloud"])
        plt.title(f"Cloud Coverage - {strategy} - {season}")
        plt.xlabel("Time (hours)")
        plt.ylabel("Cloud (0-1)")
        plt.grid()
        plt.show()

        # Inverter status (0/1)
        plt.figure(figsize=(12, 3))
        plt.plot(df["time"], df["inverter_failed"].astype(int))
        plt.title(f"Inverter Failed (1=yes) - {strategy} - {season}")
        plt.xlabel("Time (hours)")
        plt.ylabel("Failed")
        plt.grid()
        plt.show()


# Tablas finales
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

results_df = pd.DataFrame(all_results)
cloud_impact_df = pd.concat(all_cloud_impacts, ignore_index=True)

print("\n===== RESULTS (strategy × season) =====")
print(results_df)

print("\n===== Pivot: Net Cost =====")
print(results_df.pivot(index="season", columns="strategy", values="net_cost"))

print("\n===== Pivot: Grid Import (kWh) =====")
print(results_df.pivot(index="season", columns="strategy", values="grid_import_kwh"))

print("\n===== Pivot: Avg SoC (kWh) =====")
print(results_df.pivot(index="season", columns="strategy", values="avg_soc_kwh"))

print("\n===== Pivot: Unmet % Time (how often unmet) =====")
print(results_df.pivot(index="season", columns="strategy", values="unmet_pct_time"))

print("\n===== Pivot: Avg Production per Day (kWh/day) =====")
print(results_df.pivot(index="season", columns="strategy", values="avg_generation_kwh_per_day"))

print("\n===== Pivot: Avg Consumption per Day (kWh/day) =====")
print(results_df.pivot(index="season", columns="strategy", values="avg_consumption_kwh_per_day"))

print("\n===== GLOBAL (promedio sobre estaciones) =====")
print("\nNet cost (menor es mejor):")
print(results_df.groupby("strategy")["net_cost"].mean().sort_values())

print("\nGrid import (menor es mejor):")
print(results_df.groupby("strategy")["grid_import_kwh"].mean().sort_values())

print("\nAvg SoC (mayor es mejor):")
print(results_df.groupby("strategy")["avg_soc_kwh"].mean().sort_values(ascending=False))

print("\nUnmet % time (menor es mejor):")
print(results_df.groupby("strategy")["unmet_pct_time"].mean().sort_values())

results_df.to_csv("greengridsim_results_summary.csv", index=False)
print("\nCSV generado: greengridsim_results_summary.csv")

print("\n===== CLOUD IMPACT (promedios por bin de nube) =====")
print(cloud_impact_df)

cloud_summary = cloud_impact_df.groupby("cloud_bin", observed=False).agg({
    "generation_kwh": "mean",
    "grid_import": "mean",
    "grid_export": "mean",
    "unmet_load_kwh": "mean",
    "curtailed_kwh": "mean",
    "soc": "mean"
}).reset_index()

print("\n===== CLOUD SUMMARY (promedio global por bin) =====")
print(cloud_summary)

# Grafica simple: generacion promedio vs bin de nube
plt.figure(figsize=(10, 4))
plt.plot(cloud_summary["cloud_bin"].astype(str), cloud_summary["generation_kwh"], marker="o")
plt.title("Avg Generation (kWh/step) vs Cloud Bin (global)")
plt.xlabel("Cloud Bin")
plt.ylabel("Avg Generation (kWh/step)")
plt.grid()
plt.show()
