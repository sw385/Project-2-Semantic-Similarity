from pyspark import SparkContext

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


def phaseThree(data, sc):
    print ("phase 3 ----------------------------------------------")

def main():

    # initialization
    sc = SparkContext("local", "project")
    file = "project2_test.txt"
    data = sc.textFile(file)

    # first phase
    data_one = phaseOne(data, sc)

    # second phase (TF)
    data_two = phaseTwo(data_one, sc)

    # third phase (TF-IDF)
    data_three = phaseThree(data_two, sc)

if __name__ == "__main__":
  main()
