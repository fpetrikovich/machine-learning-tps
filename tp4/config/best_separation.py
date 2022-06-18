class BestSeparation(object):
    active = False
    df_train = None
    df_test = None
    best_accuracy = 0
    cross_k = None

    def isEnabled():
        return BestSeparation.active

    def setActive(is_active):
        BestSeparation.active = is_active

    # Accuracy
    def setBestAccuracy(new_best_accuracy):
        BestSeparation.best_accuracy = new_best_accuracy

    def getBestAccuracy():
        return BestSeparation.best_accuracy
    
    # CrossK
    def setCrossK(k):
        BestSeparation.cross_k = k

    def getCrossK():
        return BestSeparation.cross_k

    # Dfs
    def setDfs(train, test):
        BestSeparation.df_train = train
        BestSeparation.df_test = test

    def getDfs():
        return BestSeparation.df_train, BestSeparation.df_test
