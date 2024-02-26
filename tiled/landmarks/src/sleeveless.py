    # import module your need
import pandas as pd
import math



def csv_to_dict(path):
    df=pd.read_csv(path)
    key=[]
    value=[]
    for i in df["image_id"]: #“score”用作值
        key.append(i)
    for j in df["landmarks"]: #“score”用作值
        value.append(eval(j))
    r = zip(key, value)
    return r



if __name__ == '__main__':
    aaa = csv_to_dict("D://downloads//Edge//csv_data//results_0//train_cloth_result.csv")
    # print(type(aaa))
    li = []
    for l in aaa:
        a = math.sqrt((l[1][6][0] - l[1][9][0]) ** 2 + (l[1][6][1] - l[1][9][1]) ** 2)
        b = math.sqrt((l[1][6][0] - l[1][11][0]) ** 2 + (l[1][6][1] - l[1][11][1]) ** 2)
        c = math.sqrt((l[1][8][0] - l[1][10][0]) ** 2 + (l[1][8][1] - l[1][10][1]) ** 2)
        d = math.sqrt((l[1][8][0] - l[1][12][0]) ** 2 + (l[1][8][1] - l[1][12][1]) ** 2)
        if a <=14 or b <=14 or c <=14 or d <=14:
            li.append(l[0])
    # print(li)
    bbb = csv_to_dict("D://downloads//Edge//csv_data//results_0//train_img_result.csv")
    ccc = dict(bbb)
    # print(ccc)
    for i in li:
        del ccc[i]
    for i in ccc.items():
        data = {"name": i[0], "type": [i[1]]}
        frame = pd.DataFrame(data)
        frame.to_csv("D:/Fashion AI-keypoints/test/test/test_123.csv", mode="a", index=False, header=False)
        # print(frame)