import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df_expert = pd.read_csv("altitude_stats_expert.csv")
df_fail = pd.read_csv("altitude_stats_fail.csv")

# Filter for the terminal phase (altitude <= 15m)
df_exp_term = df_expert[df_expert['altitude_bin_center_m'] <= 15.0].copy()
df_fail_term = df_fail[df_fail['altitude_bin_center_m'] <= 15.0].copy()

# Variables to plot
vars_to_plot = ['aoa_deg', 'airspeed_mps', 'descent_rate_mps', 'pitch_deg', 'throttle_output']

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for i, var in enumerate(vars_to_plot):
    ax = axes[i]
    
    exp_var = df_exp_term[df_exp_term['variable'] == var].sort_values('altitude_bin_center_m')
    fail_var = df_fail_term[df_fail_term['variable'] == var].sort_values('altitude_bin_center_m')
    
    ax.plot(exp_var['altitude_bin_center_m'], exp_var['mean'], 'g-', label='Expert (FBWA)', marker='o')
    ax.fill_between(exp_var['altitude_bin_center_m'], 
                    exp_var['mean'] - exp_var['std_between_landings'], 
                    exp_var['mean'] + exp_var['std_between_landings'], color='green', alpha=0.2)
                    
    ax.plot(fail_var['altitude_bin_center_m'], fail_var['mean'], 'r-', label='Fail (AUTO)', marker='x')
    ax.fill_between(fail_var['altitude_bin_center_m'], 
                    fail_var['mean'] - fail_var['std_between_landings'], 
                    fail_var['mean'] + fail_var['std_between_landings'], color='red', alpha=0.2)
    
    ax.set_title(var)
    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel('Mean Value')
    ax.invert_xaxis() # Read left to right as altitude decreases
    ax.grid(True)
    ax.legend()

# Hide the empty 6th subplot
axes[5].axis('off')

plt.tight_layout()
plt.show()