class Configuration():
    verbose = False
    veryVerbose = False

    def setVerbose(is_verbose):
        Configuration.verbose = is_verbose

    def isVerbose():
        return Configuration.verbose
    
    def setVeryVerbose(is_verbose):
        Configuration.veryVerbose = is_verbose

    def isVeryVerbose():
        return Configuration.veryVerbose
