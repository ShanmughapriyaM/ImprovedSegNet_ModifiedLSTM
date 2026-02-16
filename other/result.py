import pandas as pd
import numpy as np
from colorama import Fore, init
import matplotlib.pyplot as plt
init(autoreset=True)


def stat_analysis(xx):
    mn = np.mean(xx, axis=0).reshape(-1, 1)
    mdn = np.median(xx, axis=0).reshape(-1, 1)
    std_dev = np.std(xx, axis=0).reshape(-1, 1)
    min = np.min(xx, axis=0).reshape(-1, 1)
    mx = np.max(xx, axis=0).reshape(-1, 1)
    return np.concatenate((mn, mdn, std_dev, min, mx), axis=1)


def result(db):
    column = ['LSTM ', 'DEEP MAXOUT ', 'DNN', 'SVM', 'KNN', 'LinkNet', 'BiLSTM', 'Improved Bi-LSTM+LinkNet']
    plot_result = pd.read_csv(f'pre_evaluated/saved/Comparision{db}.csv', index_col=[0, 1])
    indx = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr', 'fdr']

    avg = plot_result.loc[90, :]
    avg.reset_index(drop=True, level=0)
    avg.to_csv(f'Results/Performance Analysis{db}.csv')
    print('\n\t', Fore.LIGHTBLUE_EX + f'Performance Analysis{db}')
    print(avg.to_markdown())

    print('\n\t', Fore.LIGHTBLUE_EX + f'Ablation Study{db}')
    anal = pd.read_csv(f'pre_evaluated/saved/Ablation Study{db}.csv', index_col=[0], header=[0])
    print(pd.read_csv(f'pre_evaluated/saved/Ablation Study{db}.csv', index_col=[0], header=[0]).to_markdown())
    anal.to_csv(f'Results/Ablation Study{db}.csv')

    print('\n\t', Fore.LIGHTBLUE_EX + f'Statistics analysis{db}')
    state = pd.read_csv(f'pre_evaluated/saved/statistics analysis{db}.csv', header=0, names=column)
    print(pd.read_csv(f'pre_evaluated/saved/statistics analysis{db}.csv', header=0, names=column).to_markdown())
    state.to_csv(f'Results/Statistics analysis{db}.csv')

    # Analysis plot
    for idx, jj in enumerate(indx):
        colors = ['#669999', 'lime', '#0040ff', '#ffff00', '#4d4d4d', '#9900ff', '#00ffff', '#ff0000']
        new_ = plot_result.loc[([60, 70, 80, 90], [jj]), :]
        new_.reset_index(drop=True, level=1, inplace=True)
        new = new_.values
        br1 = np.arange(4)
        plt.style.use('grayscale')
        plt.figure(figsize=(12, 7))
        for i in range(new.shape[1]):
            plt.bar(br1, new[:, i], color=colors[i], width=0.085,
                edgecolor='k', label=column[i])
            br1 = [x + 0.085 for x in br1]
        plt.subplots_adjust(bottom=0.2)
        plt.grid(color='g', linestyle=':', linewidth=0.9)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=12, ncol=5)
        plt.xlabel('Training Data (%)', weight='bold', size=17)
        plt.ylabel(jj.upper(), weight='bold', size=17)
        plt.xticks([r for r in range(4)],
               ['60', '70', '80', '90'])
        plt.savefig(f'Results/Dataset{db}-{jj.upper()}.png', dpi=800)
    plt.show()
