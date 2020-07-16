import codecs

def rounder(num):
    return round(num, 2)


def eval_multi_acc_file(run_file, ref_file):
    run_dict = {}
    with codecs.open(run_file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t',3)
            assert len(temp) == 4
            run_dict[temp[1]] = temp[2]
    ref_dict = {}

    with codecs.open(ref_file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t',3)
            #print(temp)
            assert len(temp) == 4

            if temp[1] in ref_dict:
                ref_dict[temp[1]].append(temp[2])  # [text1(,text2)]
            else:
                ref_dict[temp[1]] = [temp[2]]  # [text1]


        t = 0.
    for id in run_dict:
        #print(run_dict[id])
        #print(ref_dict[id])
        #print("\n")
        if run_dict[id] in ref_dict[id]:
            t+=1

    return {'Multi_Acc': rounder(t*100/len(run_dict))}

