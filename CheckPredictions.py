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

from collections import Counter
predictions = []
with open('E:/THESIS/scibert/anlp_rbert/eval/biobertFINAL.txt') as f:
            for l in f.readlines():
                predictions.append(l.split('	')[1].strip())


with open('E:/THESIS/scibert/anlp_rbert/test.tsv', encoding="utf8") as f:
  correct = set()
  i = 0
  for l in f.readlines():
    if RELATIONZ[predictions[i]] in l[0:]:
        print(predictions[i])
        print(l[0:])
        correct.add((l,predictions[i]))
        counter = Counter([x[1] for x in correct])
        print(counter)
        #print('\n')

    i+=1







