import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Streamlit App Title
st.title("Pandemic SEIR Model Simulation")

# User Input Parameters
N = st.number_input("Total Population", min_value=1000, max_value=10000000, value=1000000, step=10000)
beta = st.slider("Transmission Rate (Beta)", min_value=0.1, max_value=1.0, value=0.3, step=0.01)
sigma = 1 / st.slider("Incubation Period (Days)", min_value=1, max_value=14, value=5, step=1)
gamma = 1 / st.slider("Infectious Period (Days)", min_value=1, max_value=21, value=10, step=1)
fatality_rate = st.slider("Fatality Rate", min_value=0.0, max_value=0.1, value=0.02, step=0.001)
initial_infected = st.number_input("Initial Infected Individuals", min_value=1, max_value=1000, value=10)
initial_exposed = st.number_input("Initial Exposed Individuals", min_value=1, max_value=1000, value=5)
days = st.slider("Simulation Days", min_value=30, max_value=365, value=180, step=10)

# Initial conditions
S0 = N - initial_infected - initial_exposed
E0 = initial_exposed
I0 = initial_infected
R0 = 0
D0 = 0

def seir_model(t, y):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * (1 - fatality_rate) * I
    dDdt = gamma * fatality_rate * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

# Solve the SEIR model
solution = solve_ivp(seir_model, [0, days], [S0, E0, I0, R0, D0], t_eval=np.arange(0, days, 1))
S, E, I, R, D = solution.y
time = solution.t

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time, S, label="Susceptible")
ax.plot(time, E, label="Exposed")
ax.plot(time, I, label="Infected", color='red')
ax.plot(time, R, label="Recovered", color='green')
ax.plot(time, D, label="Dead", color='black')
ax.set_xlabel("Days")
ax.set_ylabel("Population")
ax.set_title("SEIR Model Pandemic Simulation")
ax.legend()
ax.grid()
st.pyplot(fig)

# Display peak infection data
st.write("## Peak Infection Data")
st.write(f"Peak Infected: {int(max(I))} individuals")
st.write(f"Total Deaths: {int(D[-1])} individuals")
