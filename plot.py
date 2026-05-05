import matplotlib.pyplot as plt
import pandas as pd
import os

params = {
            'text.usetex' : True,
            'font.family' : 'serif',
            'font.size' : 12,
            'text.latex.preamble' : '\n'.join([
                    r'\usepackage{amsfonts}',
                ]),
}
plt.rcParams.update(params)

path = os.path.abspath(os.path.dirname(__file__))

data = pd.read_csv(path + '/results.csv')

fig, ax = plt.subplots(figsize=(3.38, 2.3), layout='tight')

plt.hlines(y=0, xmin=0, xmax=4, colors='black', linewidth=1)

ax.plot(data['Beta'], data['Avg_Current'], linestyle='-', label='$J$')
ax.plot(data['Beta'], data['Cov_B1'], linestyle='--', label=r'Cov$(J, \Lambda)$')
ax.plot(data['Beta'], data['Cov_B2'], linestyle='-.', label=r'Cov$(J, B_2)$')

ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'average value')
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_yticks([-0.05, 0, 0.05, 0.10])
ax.set_xlim(0, 4)

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='center right', bbox_to_anchor=(1.8, 0.5), fontsize=12, ncol=1, frameon=False)

# plt.legend(bbox_to_anchor=(0.4, 1.3), loc='upper center', fontsize=10, ncol=3, frameon=False)

# plt.savefig(path + '/current_response.pdf')
fig.savefig(path + '/samplefigure.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.grid()
plt.show()