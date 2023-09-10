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
    #ax.set_ylim(bottom=0.2)
    ax.set_xlabel('Classes ', fontsize=15)
    ax.set_ylabel(metric, fontsize=18)
    plt.setp(ax.get_xticklabels(), rotation=0)
    #plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper right', prop={'size': 10}, ncol=5)
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    result_files = ['C.Elegan','ElectricPowergrid','US_Air97_','ZacharyKarateClub']#sorted(os.listdir('results/'))
    print(result_files)


    for filename  in result_files:
        dfs_list = defaultdict(list)
        datasets = []
        for i in range(4):
            df = pd.read_csv('ResultNew/'+filename+str(i)+'.csv')
            mod_values = list(df['Modularity '].iloc[[0,2,4,6,8]].values)
            no_com_values = list(df['No communitiesof comunities'].iloc[[0,2,4,6,8]].values)

            dfs_list['Modularity'].append(mod_values)
            dfs_list['No_community'].append(no_com_values)

        datasets = ['Class_00', 'Class_01', 'Class_02', 'Class_03']
        x_names = ['Iteration_0.1', 'Iteration_0.3', 'Iteration_0.5', 'Iteration_0.7', 'Iteration_0.9']

        for metric in ['Modularity', 'No_community']:
            print(len(dfs_list[metric]), len(datasets))

            new_array = np.transpose(np.array(dfs_list[metric]))
            print(new_array.shape)
            df = pd.DataFrame(new_array, columns=datasets, index=x_names)
            print(df.head(8))

            visualize(new_array, datasets, x_names, 'new_results/'+filename+'_'+metric+'.png', breadth=18, metric=metric)


    '''
    datasets = [filename[:-4] for filename in result_files]
    x_names = ['CDMIILG_00', 'CDMIILG_01', 'SVM_00', 'SVM_01', 'DT_00', 'DT_01', 'RF_00', 'RF_01']

    for metric in ['Precision', 'Recall', 'F1-score', 'Accuracy']:
        print(len(dfs_list[metric]), len(datasets))

        new_array = np.transpose(np.array(dfs_list[metric]))
        print(new_array.shape)
        df = pd.DataFrame(new_array, columns=datasets, index=x_names)
        print(df.head(8))

        #visualize(new_array, datasets, x_names, metric+'.png', breadth=18, metric=metric)
    '''

