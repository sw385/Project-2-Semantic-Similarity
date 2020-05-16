from pyspark import SparkContext
from math import log

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
    return reduced2

def main():

    # initialization
    sc = SparkContext("local", "project")
    file = "project2_test.txt"
    data = sc.textFile(file)
    num_of_documents = data.count()

    # first phase
    data_one = phaseOne(data, sc)

    # second phase (TF)
    data_two = phaseTwo(data_one, sc)

    # third phase (TF-IDF)
    data_three = phaseThree(data_two, sc, num_of_documents)

    # ((document_id, term), tf*idf)
    term_tfidf = data_three

if __name__ == "__main__":
  main()
