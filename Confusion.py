RELATIONZ = {'causes1-causes2(e1,e2)': '1	causes1	causes2	1',
                     'causes2-causes1(e2,e1)': '2	causes2	causes1	2',
                     'contraindicates1-contraindicates2(e1,e2)': '3	contraindicates1	contraindicates2	1',
                     'contraindicates2-contraindicates1(e2,e1)': '4	contraindicates2	contraindicates1	2',
                     'location1-location2(e1,e2)': '5	location1	location2	1',
                     'location2-location1(e2,e1)': '6	location2	location1	2',
                     'treats1-treats2(e1,e2)': '7	treats1	treats2	1',
                     'treats2-treats1(e2,e1)': '8	treats2	treats1	2',
                     'diagnosed by1-diagnosed by2(e1,e2)': '9	diagnosed by1	diagnosed by2	1',
                     'diagnosed by2-diagnosed by1(e2,e1)': '10	diagnosed by2	diagnosed by1	2'}


predictions = []
with open('E:/THESIS/scibert/anlp_rbert/eval/FIANLBERT.txt') as f:
            for l in f.readlines():
                predictions.append(l.split('	')[1].strip())

from sklearn.metrics import multilabel_confusion_matrix

# Grab the true predictions:

truths = []
with open('E:/THESIS/scibert/anlp_rbert/test.tsv') as f:
    for l in f.readlines():
      found = False
      for k,v in RELATIONZ.items():
        if v in l:
          truths.append(k)

confusion = multilabel_confusion_matrix(truths, predictions)

for i, c in enumerate(confusion):
    print(sorted(list(RELATIONZ.keys()))[i])
    print(c)

from sklearn.metrics import precision_recall_fscore_support

precision,recall,_,_ = precision_recall_fscore_support(truths,predictions)


print('Precision    ', 'Recall')
print()
for i,p in enumerate(precision):
  print(sorted(list(RELATIONZ.keys()))[i])
  print('%.4f' % p, '      ', '%.4f' % recall[i])
  print()