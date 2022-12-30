import functools


def checkInDatabase(charList, databaseName):
    if len(charList) <= 3:
        return False
    databaseFile = open(databaseName, "r", encoding='utf-8')
    charListLen = len(charList)
    for line in databaseFile:
        currLine = line.split()
        if functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, currLine, charList), True):
            return True
    return False
