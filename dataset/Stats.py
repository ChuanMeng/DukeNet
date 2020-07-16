import codecs
import nltk

def ngram(words, n):
    ngrams=set()
    for i in range(len(words)):
        if i+n <= len(words):
            ngrams.add(' '.join(words[i:i+n]))
    return ngrams

passages=dict()
samples=[]

with codecs.open('../datasets/marco/marco.passage', encoding='utf-8') as f:
    for line in f:
        temp = line.strip('\n').strip('\r').split('\t')
        if len(temp)==2:
            passages[temp[0]]=temp[1]

with codecs.open('../datasets/marco/marco.answer', encoding='utf-8') as f:
    for line in f:
        temp = line.strip('\n').strip('\r').split('\t')
        if len(temp)!=4:
            continue
        if temp[2] in passages:
            samples.append([' '.join(nltk.word_tokenize(passages[temp[2]].lower())), nltk.word_tokenize(temp[3].lower())])

for n in range(4):
    ratio=0.
    drop=0
    for s in samples:
        words=ngram(s[1], n+1)
        if len(words)<1:
            drop+=1
            continue
        count=0
        for w in words:
            if w in s[0]:
                count+=1
        ratio+=float(count)/len(words)
    print(n, ratio/(len(samples)-drop))

# cast
# 0 0.8204028787322551
# 1 0.5144905172707461
# 2 0.390762955291243
# 3 0.32134556986842866

# marco
# 0 0.9093237364915682
# 1 0.72910424419212
# 2 0.6342457377640226
# 3 0.5800839843189012

# support sentence num, answer平均长度，conversational words比例， perlexity (BERT)