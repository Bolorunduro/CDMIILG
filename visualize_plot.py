import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import numpy as np

def visualize(input_matrix, langs, cols, filename, breadth=15, legend_loc='upper left', metric='Precision'):

    print(input_matrix.shape, len(cols))
    df = pd.DataFrame(input_matrix, columns=langs)
    ynames = cols
    xnames = langs

    all_res =  np.transpose(df.values)
    #all_res = df.values

    res_df = pd.DataFrame(all_res,
                          index=xnames,
                          columns=pd.Index(ynames,
                                           name='Models ')).round(2)
    #print(res_df.columns)
    #res_df.columns = ['Models ']+langs

    # for the Baseline model comparison: XLM-R, AfroXLMR, mDEBERTa, switched colors
    colors = ['#a6cee3', '#1f78b4', '#b2182b', '#d6604d', '#f4a582', '#b2df8a', '#33a02c', '#984ea3']


    print(res_df.index)
    res_df.plot(kind='bar', figsize=(breadth, 5), width=0.5, color=colors)

    ax = plt.gca()
    pos = []
    for bar in ax.patches:
        pos.append(bar.get_x() + bar.get_width()  / 2.)


    ax.set_xticks(pos, minor=True)
    ax.tick_params(axis='x', which='major', pad=15, size=0)

    #for bars in ax.containers:
    #   ax.bar_label(bars, padding=1)

    plt.legend(loc=legend_loc, prop={'size': 7})
    ax.set_ylim(bottom=0.8)
    ax.set_xlabel('Datasets ', fontsize=15)
    ax.set_ylabel(metric, fontsize=18)
    plt.setp(ax.get_xticklabels(), rotation=0)
    #plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper left', prop={'size': 16}, ncol=5)
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    result_files = sorted(os.listdir('results/'))
    print(result_files)

    dfs_list = defaultdict(list)
    for filename  in result_files:
        df = pd.read_csv('results/'+filename)
        prec_values = list(df['Precision'].values)
        recall_values = list(df['Recall'].values)
        f1_values = list(df['F1-Score'].values)
        acc_values = list(df['Accuracy'].values)

        dfs_list['Precision'].append(prec_values)
        dfs_list['Recall'].append(recall_values)
        dfs_list['F1-score'].append(f1_values)
        dfs_list['Accuracy'].append(acc_values)


    datasets = [filename[:-4] for filename in result_files]
    x_names = ['CDMIILG_00', 'CDMIILG_01', 'SVM_00', 'SVM_01', 'DT_00', 'DT_01', 'RF_00', 'RF_01']

    for metric in ['Precision', 'Recall', 'F1-score', 'Accuracy']:
        print(len(dfs_list[metric]), len(datasets))

        new_array = np.transpose(np.array(dfs_list[metric]))
        print(new_array.shape)
        df = pd.DataFrame(new_array, columns=datasets, index=x_names)
        print(df.head(8))

        visualize(new_array, datasets, x_names, metric+'.png', breadth=18, metric=metric)


        '''
        for dataname in datasets:
            plt.plot(df.index, df[dataname], marker='.', linestyle='none')
        plt.xlabel("Method")
        plt.ylabel(metric)
        plt.title('multiple plots')
        plt.savefig('results.png', bbox_inches='tight')
        #plt.show()
        '''

    #visualize_plot('data/result_files/LAFAND_swa_m2m_n.tsv', 'data/result_plots/lafand_sw.png', en_xx=15.9, xx_en=20.7, breadth=10)

