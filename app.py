from pyspark import SparkContext

def phaseOne(data, sc):
    print ("phase 1 ----------------------------------------------")
    results = [] # list of pairs
    split = data.map(lambda x: x.split()).collect()
    for x in split:
        document_id = x.pop(0)
        for y in x:
            pair = (document_id, y) # (document_id, term)
            results.append(pair)
    phaseOne = sc.parallelize(results)
    mapped = phaseOne.map(lambda x: (x, 1)) # ((document_id, term), 1)
    occurrences = mapped.groupByKey().mapValues(len) # ((document_id, term, occurrences))
    return occurrences


def phaseTwo(data, sc):
    print ("phase 2 ----------------------------------------------")

def phaseThree(data, sc):
    print ("phase 3 ----------------------------------------------")

def main():

    # initialization
    sc = SparkContext("local", "project")
    file = "project2_test.txt"
    data = sc.textFile(file)
    total_num_of_documents = data.count()

    # first phase
    data_one = phaseOne(data, sc)

    # second phase (TF)
    data_two = phaseTwo(data_one, sc)

    # third phase (TF-IDF)
    data_three = phaseThree(data_two, sc)

if __name__ == "__main__":
  main()
