import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# funkcja, która tworzy dwa wykresy
# 1 - pokazuje liczbę wystąpień dla danych kategorycznych
# 2 - pokazuje jakie jest średnie prawdopodobieństwo niespłacenia kredytu w konkretnej grupie
def plot_stats(df, feature,label_rotation=False,horizontal_layout=True):
    temp = df[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    cat_perc = df[[feature, 'target']].groupby([feature],as_index=False).mean()
    cat_perc['target'] = cat_perc['target'].apply(lambda x: x * 100)
    # cat_perc.sort_values(by='target', ascending=False, inplace=True)
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Number of contracts",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y='target', order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show()

def draw_boxplots(data, features, target_column):
    num_features = len(features)
    plt.figure(figsize=(18, num_features * 6))

    for i, feature in enumerate(features, 1):
        plt.subplot(num_features, 2, i)
        sns.boxplot(x=target_column, y=feature, data=data, hue=target_column, palette='pastel')
        plt.xlabel('Default status')  # 1-loan went into default, 0-facility performing
        plt.ylabel(feature)
        plt.title(f'Relationship between {feature} and default status')

    plt.tight_layout()
    plt.show()

