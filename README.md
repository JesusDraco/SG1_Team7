# SG1_Team7
Green grid simulation in python

Green Grid Simulation is a discrete-event simulation of a residential solar–battery–grid energy system built with SimPy.

This project models:

- Solar generation affected by cloud coverage  
- Battery storage with round-trip efficiency  
- Grid import/export limits and pricing  
- Inverter failures and downtime (MTTF-based model)  
- Dynamic household load  

The system acts as a **Digital Twin**, replicating the behavior of a real residential energy setup to evaluate performance, reliability, and economic impact.

---

## Energy Management Strategies

The simulation compares three strategies:

- **LOAD_PRIORITY** – Solar → Load → Battery → Grid  
- **CHARGE_PRIORITY** – Solar → Battery → Load → Grid  
- **PRODUCE_PRIORITY** – Solar → Grid → Battery → Load  

---

## Metrics Computed

- Total generation and consumption (kWh)  
- Grid import/export (kWh)  
- Curtailed energy  
- Unmet load (total, events, percentage of time)  
- Battery statistics (SoC, efficiency)  
- Inverter failures and downtime  
- Net economic cost  

---

## Outputs

The simulation generates:

- Time-series performance plots  
- Hourly CSV summaries  
- Global results summary CSV  

---

## Requirements

Install dependencies:

```bash
pip install simpy pandas matplotlib
```

---

## Run

```bash
python greengridsim.py
```

---
