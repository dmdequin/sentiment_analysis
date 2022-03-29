import json
inputPath = '../data/predictions/group12.json'

def thing():
    for lineIdx, line in enumerate(open(inputPath)):
        try:
            data = json.loads(line)
        except ValueError as e:
            print('error, instance ' + str(lineIdx+1) + ' is not in valid json format')
            continue
        if 'reviewText' not in data:
            print("error, instance " + str(lineIdx+1) + ' does not contain key "reviewText"')
            continue
        if 'sentiment' not in data:
            print("error, instance " + str(lineIdx+1) + ' does not contain key "sentiment"')
            continue
        if data['sentiment'] not in ['positive', 'negative']:
            print("error, instance " + str(lineIdx+1) + ': sentiment is not positive/negative')
            continue
            
    if lineIdx+1 < 100:
        print('Too little instances(' + str(lineIdx) + '), please generate more')
    if lineIdx+1 > 1000:
        print('Too many instances(' + str(lineIdx) + '), please generate more')


if __name__ == '__main__':
    thing()