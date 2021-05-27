import sys
from indicnlp.tokenize import indic_tokenize  

hypothesis = sys.argv[-1]
reference = sys.argv[-2]

#Reading the file as list of sentences
with open(hypothesis) as file_in:
    hypothesis = []
    for line in file_in:
        hypothesis.append(line)
        
with open(reference) as file_in:
    reference = []
    for line in file_in:
         reference.append(line)
            
assert len(hypothesis) == len(reference)   

#Hindi Language tokenization
hypothesis_tok = [' '.join(indic_tokenize.trivial_tokenize(item[:-1])) for item in  hypothesis]
reference_tok = [' '.join(indic_tokenize.trivial_tokenize(item[:-1])) for item in  reference]
assert len(hypothesis_tok) == len(reference_tok)

with open('outputs/hindi_hyp.txt', 'w') as f:
    for item in hypothesis_tok:
        f.write("%s\n" % item)
        
with open('outputs/hindi_ref.txt', 'w') as f:
    for item in reference_tok:
        f.write("%s\n" % item)         
