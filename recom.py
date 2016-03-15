from pyspark.mllib.recommendation import *
import random
import code
from operator import *

def canonical(inp):
    if aliasDict.has_key(inp[1]):
        return (inp[0], aliasDict.get(inp[1]), inp[2])
    else:
        return inp

artistData = sc.textFile("artist_data_small.txt").map(lambda x: (x.split("\t")[0], x.split("\t")[1]))
artistAlias = sc.textFile("artist_alias_small.txt")
aliasDict = artistAlias.map(lambda x: (x.split("\t")[0], (x.split("\t")[1]))).collectAsMap()
userArtistData = sc.textFile("user_artist_data_small.txt").map(lambda x : (x.split(" ")[0], x.split(" ")[1], x.split(" ")[2]))
userArtistData = userArtistData.map(canonical)

def meanUser(inp):
    return (str(inp[0]), str(inp[1]), (int(inp[1]) / int(count.get(inp[0]))))
userCount = userArtistData.map(lambda x: (x[0],x[2])).reduceByKey(lambda x, y: (int(x) + int(y))).sortBy(lambda x : x[1], ascending = False)
count = userArtistData.countByKey()
userCountMean = userCount.map(meanUser).collect()
#print "User " + str(userCountMean[0][0]) + " has a total play count of " + str(userCountMean[0][1]) + " and a mean play count of " + str(userCountMean[0][2])
#print "User " + str(userCountMean[1][0]) + " has a total play count of " + str(userCountMean[1][1]) + " and a mean play count of " + str(userCountMean[1][2])
#print "User " + str(userCountMean[2][0]) + " has a total play count of " + str(userCountMean[2][1]) + " and a mean play count of " + str(userCountMean[2][2])

userArtistData = userArtistData.map(lambda x: (int(x[0]), int(x[1]), int(x[2])))
trainData, validationData, testData = userArtistData.randomSplit([40,40,20], seed = 13)
#print trainData.take(3)
#print validationData.take(3)
#print testData.take(3)
#print len(trainData.collect())
#print len(validationData.collect())
#print len(testData.collect())

allArtists = userArtistData.map(lambda x : x[1]).distinct()
#def modelEval(model, data):
data = validationData
model = ALS.trainImplicit(trainData, rank=2, seed=345)
#finalResult = 0
users = data.map(lambda x : x[0]).collect()
#for user in users:
userArtists = trainData.filter(lambda x : x[0] == user).map(lambda x: x[1])
nonTrainArtists = allArtists.filter(lambda x : x[0] not in userArtists).map(lambda x : x[1])
trueArtists = data.filter(lambda x : x[0] == user).map(lambda x : x[1])
x = len(trueArtists.collect())
userNonTrainArtists = nonTrainArtists.map(lambda x : (user, x))
code.interact(local = locals())
print userNonTrainArtists.take(3)
predictResult = model.predictAll(userNonTrainArtists)
print predictResult.take(3)
comparison = trueArtists.interesection(predictResult)
complen = comparison.collect()
finalResult += float(len(complen))/float(x)
#return finalResult