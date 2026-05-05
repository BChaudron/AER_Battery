from os import path
import pyomo.environ as pyo
import pandas as pd
import highspy

from init import FILE_1, FILE_2, OUTPUT_FOLDER


# Data
print("loading data...")

df_1 = pd.read_excel(FILE_1, sheet_name="Data")
df_2a = pd.read_excel(FILE_2, sheet_name="Half-hourly data")
df_2b = pd.read_excel(FILE_2, sheet_name="Daily data")

prices_mkt_1 = df_2a["Market 1 Price [£/MWh]"]
prices_mkt_2 = df_2a["Market 2 Price [£/MWh]"]
prices_mkt_3 = df_2b["Market 3 Price [£/MWh]"]

time_index = pd.to_datetime(df_2a.iloc[:, 0])

T = list(range(len(time_index)))
days = sorted({ts.date() for ts in time_index})
D = list(range(len(days)))

day_of_t = {t: days.index(time_index[t].date()) for t in T }

T_in_D = {
    d: [t for t in T if day_of_t[t] == d]
    for d in D
}

P_CH_MAX = df_1[df_1.iloc[:, 0] == "Max charging rate"]["Values"].values[0]
P_DIS_MAX = df_1[df_1.iloc[:, 0] == "Max discharging rate"]["Values"].values[0]
E_MAX = df_1[df_1.iloc[:, 0] == "Max storage volume"]["Values"].values[0]/2 # max storage volume in half-hourly hours (since we multiply by DT in the constraints)
ETA_C = 1-df_1[df_1.iloc[:, 0] == "Battery charging efficiency"]["Values"].values[0]
ETA_D = 1-df_1[df_1.iloc[:, 0] == "Battery discharging efficiency"]["Values"].values[0]
E_INIT = df_1[df_1.iloc[:, 0] == "Max storage volume"]["Values"].values[0]/2  # Start at 50% SOC
DT = 0.5 # hours (half-hourly data)


# Model

m = pyo.ConcreteModel()

m.T = pyo.Set(initialize=T)
m.D = pyo.Set(initialize=D)
m.T_in_D = pyo.Set(m.D, initialize=T_in_D, within=m.T) # for every day d, define the subset of half‑hour time steps that belong to that day


m.p1_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals)
m.p1_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals)

m.p2_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals)
m.p2_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals)

m.p3_ch = pyo.Var(m.D, domain=pyo.NonNegativeReals)
m.p3_dis = pyo.Var(m.D, domain=pyo.NonNegativeReals)

m.e = pyo.Var(m.T, domain=pyo.NonNegativeReals)


# Objective

def profit_rule(m):
    revenue = 0

    for t in m.T:
        revenue += (
                           prices_mkt_1[t] * (m.p1_dis[t] - m.p1_ch[t])
                           + prices_mkt_2[t] * (m.p2_dis[t] - m.p2_ch[t])
                   ) * DT

    for d in m.D:
        revenue += prices_mkt_3[d] * (m.p3_dis[d] - m.p3_ch[d]) * 24

    return revenue


m.obj = pyo.Objective(rule=profit_rule, sense=pyo.maximize)


# Constraints

m.charge_limit = pyo.Constraint(
    m.T, rule=lambda m, t: (m.p1_ch[t] + m.p2_ch[t] + m.p3_ch[day_of_t[t]]) <= P_CH_MAX
)
m.discharge_limit = pyo.Constraint(
    m.T, rule=lambda m, t: (m.p1_dis[t] + m.p2_dis[t] + m.p3_dis[day_of_t[t]]) <= P_DIS_MAX
)

def soc_rule(m, t):
    if t == 0:
        return m.e[t] == E_INIT \
               + ETA_C * (m.p1_ch[t] + m.p2_ch[t] + m.p3_ch[day_of_t[t]]) * DT \
               - (1 / ETA_D) * (m.p1_dis[t] + m.p2_dis[t] + m.p3_dis[day_of_t[t]]) * DT
    return m.e[t] == m.e[t-1] \
                     + ETA_C * (m.p1_ch[t] + m.p2_ch[t] + m.p3_ch[day_of_t[t]]) * DT \
                     - (1 / ETA_D) * (m.p1_dis[t] + m.p2_dis[t] + m.p3_dis[day_of_t[t]]) * DT


m.soc = pyo.Constraint(m.T, rule=soc_rule)

m.energy_limit = pyo.Constraint(
    m.T, rule=lambda m, t: m.e[t] <= E_MAX
)


# Solve
print("solving model...")

solver = pyo.SolverFactory("highs")
solver.solve(m)


# Results

print("post-processing results...")

results = pd.DataFrame({
    "Date" : time_index,
    "Price Market 1": prices_mkt_1,
    "Price Market 2": prices_mkt_2,
    "Price Market 3": prices_mkt_3.repeat(48).reset_index(drop=True),

    "Total Charge": [pyo.value(m.p1_ch[t] + m.p2_ch[t] + m.p3_ch[day_of_t[t]]) for t in m.T],
    "Total Discharge": [pyo.value(m.p1_dis[t] + m.p2_dis[t] + m.p3_dis[day_of_t[t]]) for t in m.T],
    "Total Energy Stored": [pyo.value(m.e[t]) for t in m.T],

    "Charge in Market 1": [pyo.value(m.p1_ch[t]) for t in m.T],
    "Discharge in Market 1": [pyo.value(m.p1_dis[t]) for t in m.T],
    "Charge in Market 2": [pyo.value(m.p2_ch[t]) for t in m.T],
    "Discharge in Market 2": [pyo.value(m.p2_dis[t]) for t in m.T],
    "Charge in Market 3": [pyo.value(m.p3_ch[day_of_t[t]]) for t in m.T],
    "Discharge in Market 3": [pyo.value(m.p3_dis[day_of_t[t]]) for t in m.T],
})

objective_value = pyo.value(m.obj)

# Calculate daily revenues
revenue_data = []
for d in D:
    market1_rev = sum(prices_mkt_1.iloc[t] * (pyo.value(m.p1_dis[t]) - pyo.value(m.p1_ch[t])) * DT for t in T_in_D[d])
    market2_rev = sum(prices_mkt_2.iloc[t] * (pyo.value(m.p2_dis[t]) - pyo.value(m.p2_ch[t])) * DT for t in T_in_D[d])
    market3_rev = prices_mkt_3.iloc[d] * (pyo.value(m.p3_dis[d]) - pyo.value(m.p3_ch[d])) * 24
    total_rev = market1_rev + market2_rev + market3_rev

    revenue_data.append({
        "Date": days[d],
        "Market 1 Revenue": market1_rev,
        "Market 2 Revenue": market2_rev,
        "Market 3 Revenue": market3_rev,
        "Total Revenue": total_rev
    })

revenue_df = pd.DataFrame(revenue_data)

# Add total row
revenue_df.loc[len(revenue_df)] = {
    "Date": "TOTAL",
    "Market 1 Revenue": revenue_df["Market 1 Revenue"].sum(),
    "Market 2 Revenue": revenue_df["Market 2 Revenue"].sum(),
    "Market 3 Revenue": revenue_df["Market 3 Revenue"].sum(),
    "Total Revenue": objective_value
}

# Extract year from dates and calculate yearly revenues
revenue_df_copy = revenue_df[revenue_df["Date"] != "TOTAL"].copy()
revenue_df_copy["Year"] = pd.to_datetime(revenue_df_copy["Date"]).dt.year

yearly_revenue_data = []
for year in sorted(revenue_df_copy["Year"].unique()):
    year_data = revenue_df_copy[revenue_df_copy["Year"] == year]
    yearly_revenue_data.append({
        "Year": year,
        "Market 1 Revenue": year_data["Market 1 Revenue"].sum(),
        "Market 2 Revenue": year_data["Market 2 Revenue"].sum(),
        "Market 3 Revenue": year_data["Market 3 Revenue"].sum(),
        "Total Revenue": year_data["Total Revenue"].sum()
    })

yearly_revenue_df = pd.DataFrame(yearly_revenue_data)

# Add total row
yearly_revenue_df.loc[len(yearly_revenue_df)] = {
    "Year": "TOTAL",
    "Market 1 Revenue": yearly_revenue_df["Market 1 Revenue"].sum(),
    "Market 2 Revenue": yearly_revenue_df["Market 2 Revenue"].sum(),
    "Market 3 Revenue": yearly_revenue_df["Market 3 Revenue"].sum(),
    "Total Revenue": objective_value
}

print("exporting results...")

with pd.ExcelWriter(path.join(OUTPUT_FOLDER, r"results.xlsx")) as writer:
    results.to_excel(writer, sheet_name="Results", index=False)
    revenue_df.to_excel(writer, sheet_name="Daily Revenues", index=False)
    yearly_revenue_df.to_excel(writer, sheet_name="Yearly Revenues", index=False)

print("done !")