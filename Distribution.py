from collections import Counter
import csv

predictions = []
with open('E:/THESIS/scibert/anlp_rbert/eval/FIANLBERT.txt') as f:
            for l in f.readlines():
                predictions.append(l.split('	')[1].strip())
print('\n Train Data distribution: \n')
train = open('E:/THESIS/scibert/anlp_rbert/train.tsv', encoding="utf8")
train_data = csv.reader(train, delimiter="\t")

print(Counter([(x[3],x[4]) for x in train_data]))
print('\n Train Data distribution: \n')


test = open('E:/THESIS/scibert/anlp_rbert/test.tsv', encoding="utf8")
test_data = csv.reader(test, delimiter="\t")
print(Counter([(x[3],x[4]) for x in test_data]))

print('\n Treats prediction: \n')
treats = []
with open('E:/THESIS/scibert/anlp_rbert/test.tsv') as f:
    for i, l in enumerate(f.readlines()):
        if 'treats' in l[-30:]:
            # it is a "treats" relation
            treats.append(predictions[i])
            if predictions[i][:7] == 'treats2' or predictions[i][:7] == 'treats1':
                # it is predicted to be a treats relation
                print(l)

print(Counter(treats))