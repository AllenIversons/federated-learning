import csv

def save_results(results, filename):
    """
    :param results: experiment results
    :type results: list()
    :param filename: File name to write results to
    :type filename: String
    """
    loss = zip(*[iter(results)]*1)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for experiment in loss:
            writer.writerow(experiment)

def read_results(filename):
    """
    :param filename: File name to read results from
    :type filename: String
    """
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            data.append(row)

    return data
