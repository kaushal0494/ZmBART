import pandas as pd
from fairseq.models.bart import BARTModel
import spacy
sp = spacy.load('en_core_web_sm')

read_data= pd.read_csv("clean_triplet_DG.csv", header=None, nrows=5)
ext_data_sent = read_data.iloc[1:, 1:2].values.tolist() 
ext_data_rest = read_data.iloc[1:, 2:].values.tolist() 
#final_sent_list = []
#for elem in ext_data_sent:
#    temp_list = []
#    for sent in sp(elem[0]).sents:
#        temp_list.append(sent.text)    
#    final_sent_list.append(temp_list)

print(ext_data_sent[0])
#print(len(final_sent_list), final_sent_list[0])
#print(len(ext_data_rest), ext_data_rest[0])
print("Reading of CSV is done!!!")

BASEDIR = '../checkpoint'
bart = BARTModel.from_pretrained(
        BASEDIR,
        checkpoint_file='checkpoint_best_en-hi_parallel.pt',
        bpe='sentencepiece',
        max_tokens=1024,
        fp16=True, 
        gpu=True,
        sentencepiece_model=f'../checkpoint/sentence.bpe.model')
bart.eval()

translation_sent = bart.sample(ext_data_sent[0], beam=5)
#translation_rest = bart.sample(ext_data_rest[0], beam=5)
print(translation_sent)
#print(translation_rest)

"""print("Transaltion In the Process....!!!")
trans_list =[]
for i, item in enumerate(ext_data): 
    if i%10 == 0:
       print(i)
       trnas_temp_data = pd.DataFrame(trans_list, columns=['id', 'passage', 'question', 'answer', 'dist1', 'dist2', 'dist3'])
       trnas_temp_data.to_csv('Hindi_DG.csv', index=False)
    temp_list =[]
    temp_list.append(str(item[0]))
    temp_list.append(' '.join([bart.sample([str(sent)], beam=5)[0][:-7] for sent in item[1]]))
    temp_list.append(bart.sample([str(item[2])], beam=5)[0][:-7]) 
    temp_list.append(bart.sample([str(item[3])], beam=5)[0][:-7]) 
    temp_list.append(bart.sample([str(item[4])], beam=5)[0][:-7]) 
    temp_list.append(bart.sample([str(item[5])], beam=5)[0][:-7]) 
    temp_list.append(bart.sample([str(item[6])], beam=5)[0][:-7])
    trans_list.append(temp_list)        

trnas_data = pd.DataFrame(trans_list, columns=['id', 'passage', 'question', 'answer', 'dist1', 'dist2', 'dist3'])
trnas_data.to_csv('Hindi_DG.csv', index=False)"""

#sentence_list = ['अंततः ऊंचे पंखों वाले सीएक्स-एचएलएस (CX-HLS) बोइंग डिजाइन का 747 के लिए उपयोग नहीं किया गया, हालांकि अपनी बोली के लिए विकसित की गई उनकी प्रौद्योगिकियों का प्रभाव हुआ था। मूल डिजाइन में वायुयान का पूर्ण-लंबाई का दो-मंजिला धड़ शामिल था, जिसकी निचली मंजिल पर आठ-आठ सीटों की पंक्तियां और दो गलियारे थे तथा ऊपरी मंजिल पर सात-सात सीटों की पंक्तियां और एक गलियारा था। हालांकि, निकासी मार्गों की समस्या और सीमित माल वहन क्षमता के कारण इस विचार को, 1966 में एक व्यापक एक मंजिला डिजाइन के पक्ष में त्याग दिया गया। इसलिए कॉकपिट को संक्षिप्त दूसरे तल पर रखा गया, ताकि सामने वली नाक या शंकु में माल के लदान के लिए दरवाजा शामिल किया जा सके; डिजाइन की इस विशेषता ने ही 747 के विशिष्ट ""उभार"" को जन्म दिया।  आरंभिक मॉडलों में यह स्पष्ट नहीं था कि कॉकपिट के पीछे संदूकनुमा छोटे से स्थान का क्या किया जाए, इसलिए शुरू में इसे बिना स्थाई बैठक व्यवस्था के विश्राम कक्ष के रूप में ""चिह्नित"" किया गया।","शुरूआती 747 में, जहाँ विमानचालक बैठते हैं उसके पीछे क्या होता था?",विश्राम कक्ष']
#translation = bart.sample(sentence_list, beam=5)
#print(translation)
#breakpoint()
