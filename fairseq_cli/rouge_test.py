import sys
from rouge import FilesRouge
files_rouge = FilesRouge() 

hypothesis = sys.argv[-1]
reference = sys.argv[-2]

scores = files_rouge.get_scores(hypothesis, reference, avg=True)
print(scores)
        
