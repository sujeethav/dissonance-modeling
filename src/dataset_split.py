import json


full_dataset = json.load(open('init_multitask_data.json'))
count=0
test=[]
train=[]
dev=[]
limit=80
limit_nd=150
cnt_nd=0
for sample in full_dataset:
    diss_pairs = sample['dissonance_pairs']
    flag=0
    for pairs in diss_pairs:
        if pairs['disso_label']=='D':
            flag=1
            count+=1
            if(count<=40):
                test.append(sample)
            elif(count>40 and count<=80):
                dev.append(sample)
            else:
                train.append(sample)
            break
    if(flag==0):
        cnt_nd+=1
        if cnt_nd <= 75:
            test.append(sample)
        elif cnt_nd>75 and cnt_nd>limit_nd<150:
            dev.append(sample)
        else:
            train.append(sample)

print(len(train))
print(len(test))
print(len(dev))

with open('train_data.json', 'w+') as train_file:
    json.dump(train, train_file, indent=4)

with open('test_data.json', 'w+') as test_file:
    json.dump(test, test_file, indent=4)

with open('dev_data.json', 'w+') as dev_file:
    json.dump(dev, dev_file, indent=4)