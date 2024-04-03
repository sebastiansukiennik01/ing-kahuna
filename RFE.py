from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
def rfe(x, y):
    regressor = RandomForestRegressor(n_estimators=100, max_depth=10)

    # here we want only one final feature, we do this to produce a ranking
    n_features_to_select = 1
    rfe = RFE(regressor, n_features_to_select=n_features_to_select)
    rfe.fit(x, y)

    #===========================================================================
    # now print out the features in order of ranking
    #===========================================================================
    from operator import itemgetter
    features = x.columns.to_list()
    for x, y in (sorted(zip(rfe.ranking_ , features), key=itemgetter(0))):
        print(x, y)