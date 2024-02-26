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
    aaa = csv_to_dict("test_cloth_result.csv")
    ccc = numpy.array(aaa)
    print(len(aaa))
    ddd = torch.tensor(ccc / [768, 1024])
    print(ddd)
    print(type(ddd))