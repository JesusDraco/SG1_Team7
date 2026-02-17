# SG1_Team7
Green grid simulation in python

Green Grid Simulation is a discrete-event simulation of a residential solar–battery–grid energy system built with SimPy.

This project models:

Solar generation affected by cloud coverage

Battery storage with efficiency losses

Grid import/export limits and pricing

Inverter failures and downtime

Dynamic household load

The simulation compares different energy management strategies:

LOAD_PRIORITY

CHARGE_PRIORITY

PRODUCE_PRIORITY

It computes technical and economic metrics such as total generation, grid import/export, unmet load frequency, battery state of charge, inverter reliability, and net cost.

The model also exports hourly CSV summaries and generates performance plots for analysis.

Requirements
pip install simpy pandas matplotlib

Run
python greengridsim.py
