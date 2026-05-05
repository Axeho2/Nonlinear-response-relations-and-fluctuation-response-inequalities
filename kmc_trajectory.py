import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# parameters
B_edge = 1.2
B_in = 1.2
E_out = 0
E_in = 1
c_left = 5
c_right = 2.5

@nb.jit(nopython=True)
def get_R(n, beta):
    # Here you can define your custom rates; an example placeholder is provided
    R = np.ones((2**n, 2**n))
    for i in range(2**n):
        R[i][i] = 0
    return R

@nb.jit(nopython=True)
def traj_generator(R, p_ini):
    state = np.searchsorted(np.cumsum(p_ini), np.random.rand(), side="right")
    waiting_time = -np.log(np.random.rand()) / np.sum(R[:, state])
    while True:
        yield waiting_time, state
        probabilities = R[:, state] / np.sum(R[:, state])
        state = np.searchsorted(np.cumsum(probabilities), np.random.rand(), side="right")
        waiting_time = -np.log(np.random.rand()) / np.sum(R[:, state])

@nb.jit(nopython=True)
def get_event(n, beta, t_eq, ensembles):
    R = get_R(n, beta)
    p_ini = np.zeros(2**n)
    p_ini[0] = 1
    event = np.zeros((ensembles, 2**n, 2**n))
    t_pre_eq = 50
    for ensemble in range(ensembles):
        t = 0
        traj = traj_generator(R, p_ini)
        waiting_time, state = next(traj)
        while t < t_pre_eq + t_eq:
            prev_state = state
            dwell_time = waiting_time
            t += dwell_time
            waiting_time, state = next(traj)
            if t > t_pre_eq:
                event[ensemble][state][prev_state] += 1
                if t > t_pre_eq + t_eq:
                    event[ensemble][prev_state][prev_state] += dwell_time - (t - t_pre_eq - t_eq)
                else:
                    event[ensemble][prev_state][prev_state] += dwell_time
    return event

@nb.jit(nopython=True)
def calculate_lambda(n, beta, event, t_eq, ensembles):
    R = get_R(n, beta)
    Lambda_total = 0.0

    for ensemble in range(ensembles):
        Lambda = 0.0
        for i in range(2**n):
            for j in range(2**n):
                if i != j:
                    Lambda += event[ensemble][i][j] - R[i][j] * event[ensemble][j][j]
        Lambda_total += Lambda

    Lambda_avg = Lambda_total / (t_eq * ensembles)
    return Lambda_avg

# Example usage
n = 3
t_eq = 500
ensembles = 100000
betas = np.linspace(0.2, 1.2, num=10)
Lambda_vals = np.zeros(len(betas))

for i in range(len(betas)):
    print(f'Calculating for beta = {betas[i]:.2f}')
    events = get_event(n, betas[i], t_eq, ensembles)
    Lambda_vals[i] = calculate_lambda(n, betas[i], events, t_eq, ensembles)

# Plotting the results
fig, ax = plt.subplots()
ax.plot(betas, Lambda_vals, marker='o', linestyle='-', label='Lambda')
ax.axhline(0, color='grey', linestyle='--')
ax.set_xlabel('Beta')
ax.set_ylabel('Average Lambda')
ax.set_title('Average Lambda vs Beta')
ax.grid(True)
ax.legend()
plt.show()
