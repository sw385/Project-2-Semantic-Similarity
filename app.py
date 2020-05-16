from pyspark import SparkContext
from math import log
from sys import argv
from functools import partial

def phaseOne(data, sc):
    print ("phase 1 ----------------------------------------------")

    # map
    results = [] # list of pairs
    split = data.map(lambda x: x.split()).collect()
    for x in split:
        document_id = x.pop(0)
        for y in x:
            pair = (document_id, y) # (document_id, term)
            results.append(pair)
    phaseOne = sc.parallelize(results)
    mapped = phaseOne.map(lambda x: (x, 1)) # ((document_id, term), 1)

    # reduce
    occurrences = mapped.groupByKey().mapValues(len) # ((document_id, term), occurrences)
    return occurrences

def phaseTwo(data, sc):
    print ("phase 2 ----------------------------------------------")

    # map
    mapped = data.map(lambda x: (x[0][0], (x[0][1], x[1]))) # (document_id, (term, occurrences))
    mapped2 = mapped.groupByKey().mapValues(list).collect() # (document_id, [(term1,occurrences1), (term2, occurrences2)])

    # reduce
    results = []
    for document in mapped2:
        sum = 0
        for term in document[1]:
            sum += term[1]
        for term in document[1]:
            pair = ((document[0], term[0]), (term[1], sum))
            results.append(pair)
    reduced = sc.parallelize(results) # ((document_id, term), (term_occurrences, total_word_count_doc))
    return reduced

def phaseThree(data, sc, num_of_documents):
    print ("phase 3 ----------------------------------------------")

    # map
    mapped = data.map(lambda x: (x[0][1], (x[0][0], x[1][0], x[1][1]))) # (term, (document_id, term_occurrences, total_word_count_doc))

    #reduce
    results = []
    mapped2 = mapped.groupByKey().mapValues(list).collect() # (term, [(document_id, term_occurrences, total_word_count_doc)])
    for term in mapped2:
        num_word_in_docs = len(term[1])
        for params in term[1]:
            pair = ((params[0], term[0]), (params[1], params[2], num_word_in_docs))
            results.append(pair)
    reduced = sc.parallelize(results) # ((document_id, term), (term_occurrences, total_word_count_doc, num_word_in_all_docs))
    reduced2 = reduced.map(lambda x: ((x[0][0], x[0][1]), ((float(x[1][0])/float(x[1][1]))*(log(float(num_of_documents)/float(x[1][2]))))))
    # ((document_id, term), tf*idf)
    # print(reduced2.collect())
    return reduced2





def sim_1_map(query, element):
    # element: ((docid, term), tfidf)
    # print("semantic similarity phase 1 map --------------------------")
    # print(element)
    docid = element[0][0]
    term = element[0][1]
    tfidf = element[1]
    return (docid, (query, term, tfidf))

def sim_2_map(query_tfidfs, element):
    # element: (docid, (query, term, tfidf))
    # print("semantic similarity phase 2 map --------------------------")
    # print(element)
    term = element[1][1]
    # iterating through a list for every element = inefficient, use a dict instead
    # v1 = [f for f in query_tfidfs if f[0] == element[0]][0][1][2]
    if element[0] in query_tfidfs:
        v1 = query_tfidfs[element[0]]    # the tfidf of our query for this docid
    else:
        v1 = 0
    v2 = element[1][2]
    return (term, (v1 * v2, v1 * v1, v2 * v2))

def sim_2_red(element1, element2):
    # element: (term, (v1 * v2, v1 * v1, v2 * v2))
    # print("semantic similarity phase 2 reduce --------------------------")
    # cannot square root and multiply until the end
    numerator = element1[0] + element2[0]
    denominator1 = element1[1] + element2[1]
    denominator2 = element1[2] + element2[2]
    return (numerator, denominator1, denominator2)

def sim_3_map(element):
    # element: (term, (numerator, denominator1, denominator2))
    # output: (null, (term, semantic_similarity))
    print("semantic similarity phase 3 map --------------------------")
    numerator = element[1][0]
    denominator1 = element[1][1]
    denominator2 = element[1][2]
    try:
        semantic_similarity = numerator / ((denominator1 ** 0.5) * (denominator2 ** 0.5))
    except:
        print(element[0])
    return (None, (element[0], semantic_similarity))

'''
this step doesn't need a reduce, just turn the collection into a sorted list
def sim_4_red():
    print("semantic similarity phase 3 reduce --------------------------")
    pass
'''


    

def main():

    # initialization
    sc = SparkContext("local", "project")
    file = "project2_test.txt"
    data = sc.textFile(file)
    query = sc.broadcast(argv[1])
    # convert unicode to ascii
    data = data.map(lambda x: x.encode("ascii", "ignore"))
    num_of_documents = data.count()

    # first phase
    data_one = phaseOne(data, sc)

    # second phase (TF)
    data_two = phaseTwo(data_one, sc)

    # third phase (TF-IDF)
    data_three = phaseThree(data_two, sc, num_of_documents)

    # ((document_id, term), tf*idf)
    term_tfidf = data_three
    


    # print(term_tfidf.take(5))

    # "partial" lets us pass arguments into the passed function
    similarities = term_tfidf.map(partial(sim_1_map, query.value))
    # print(similarities.take(5))
    # print(similarities.collect())

    # this is a map step, not a reduce step
    # the issue here is that we need to pull out the query term's tf-idf in each partition
    # a straightforward reduce would not be sufficient
    # we have a collection/dataframe, but also need a particular value from that collection to be applied over the other values
    # shouldn't use broadcast, the values we need are already collected within each node
    # shouldn't use collect, since we need to "collect" for each dataframe
    # aggregate?
    # similarities.foreachPartition(function1)
    # foreachPartition will encounter issues since we're in a single node
    # print(len(similarities.collect()))
    # I don't see a way to pass a different value into each partition's function call
    
    # total = similarities.collect()
    # print(len(total))
    query_tfidfs = similarities.filter(lambda x: x[1][1] == query.value).collect()
    # print(len(query_tfidfs))
    # print(query_tfidfs[:5])
    query_tfidfs = [(f[0], f[1][2]) for f in query_tfidfs]
    # print(query_tfidfs[:5])
    # (docid, query's tfidf for that doc)
    query_tfidfs = dict(query_tfidfs)
    similarities = similarities.filter(lambda x: x[1][1] != query.value)
    # total = similarities.collect()
    # print(len(total))
    
    # for every term in the doci, calculate v1*v2, v1*v1, and v2*v2, then group by term
    similarities = similarities.map(partial(sim_2_map, query_tfidfs))
    # print(similarities.take(5))

    # sum to obtain the dot product (numerator), denominator 1, and denominator 2
    similarities = similarities.reduceByKey(sim_2_red)
    # print(similarities.take(5))
    # print(len(similarities.collect()))  # (number of unique terms + docids) - (number of lines/docids) - (1 for the query)

    # calculate the semantic similarity for each term
    similarities = similarities.map(sim_3_map)
    # no reduce necessary
    print(len(similarities.collect()))


    print('nathaniel')

    query.unpersist()

if __name__ == "__main__":
    main()
