from sys import argv

start_processing = False
done_algorithm = dict()
top_results = []

print 'r = ['
for line in open(argv[1]):
    line = line.replace('\n', '')

    if 'Sorted results' in line:
        start_processing = True
        continue
    if start_processing:
        tokens = line.split(', ')

        if len(tokens) == 7:
            algorithm = tokens[0].replace('Model=', '')
            future = tokens[3].split('=')[1]
        else:
            algorithm = tokens[0].replace('Model=', '') + ', ' + tokens[1]
            future = tokens[4].split('=')[1]

        if future not in done_algorithm:
            done_algorithm[future] = []

        if len(done_algorithm[future]) < 3 and algorithm not in done_algorithm[future]:
            done_algorithm[future].append(algorithm)
            top_results.append(line)
        print ' \'' + line + '\','
print ']'

print 'Top results run'
for top_result in top_results:
    tokens = top_result.split(', ')

    if len(tokens) == 7:
        algorithm = tokens[0].replace('Model=', '')
        past = tokens[2].split('=')[1]
        future = tokens[3].split('=')[1]
        threshold = tokens[4].split('=')[1]
    else:
        algorithm = tokens[0].replace('Model=', '') + ', ' + tokens[1]
        past = tokens[3].split('=')[1]
        future = tokens[4].split('=')[1]
        threshold = tokens[5].split('=')[1]
    print 'Algorithm=%s, Past=%s, Future=%s, Threshold=%s' % (algorithm, past, future, threshold)

print 'Top results run list'
print 'r = ['
for top_result in top_results:
    print ' \'' + top_result + '\','
print ']'
