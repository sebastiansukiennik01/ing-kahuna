import itertools
import math
import pandas as pd
# from decomposing import difference



def do_hellwig(filename: str, potential_features: list):
    df = pd.read_csv(filename)

    # # potential features
    # potential_features = ['instant', 'holiday', 'workingday',
    #         'weathersit', 'temp', 'hum', 'windspeed']

    def hellwig(corr: pd.DataFrame, comb):
        info = 0
        for feature in comb:
            d = 0
            for feature_ in comb:
                d += abs(corr[feature][feature_])
            info += (corr[feature]['cnt'])**2/d
        return info


    # getting all possible combinations of explanatory variables
    combinations = []
    for r in range(1, len(potential_features) + 1):
        for combination in itertools.combinations(potential_features, r):
            combinations.append(combination)

    assert len(combinations) == 2**len(potential_features) - 1

    bestInfo = 0
    bestComb = []
    infos = []
    features = []

    for combination in combinations:
        info = hellwig(df.corr(), combination)
        if info > bestInfo:
            bestInfo = info
            bestComb = combination
            irrelFeatures = set(potential_features).difference(combination)
        infos.append(info)
        features.append(combination)


    df = pd.DataFrame({'features': features, 'infos': infos})
            
    with open('Data\\hellwigResults.txt', 'w') as file:
        for row in df.sort_values(['infos'], ascending=False).iterrows():
            file.write(','.join(row[1]['features']))
            file.write(f"  -- {row[1]['infos']}\n")

            

    print(bestInfo)
    print(bestComb)
    print(irrelFeatures) 