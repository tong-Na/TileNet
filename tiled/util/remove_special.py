import pandas as pd


# def get_all_landmarks(path):
#     df = pd.read_csv(path)
#     value = []
#     for j in df["landmarks"]:
#         value.append(eval(j))
#     return [list(df['image_id']), value]

phase = "train"
type = "cloth"

special = pd.read_csv(f"~/try-on/tiled/type_sort/special_type_{phase}.csv")
special = dict(zip(special["image_id"], special["type"]))

full_list = pd.read_csv(f"~/try-on/data/zalando-hd-resized/{phase}/{phase}_{type}.csv")

remove_id = []
remove_points = []
for _, row in full_list.iterrows():
    name = row["image_id"].split("/")[-1]
    if special.get(name) is not None:
        continue
    remove_id.append(row["image_id"])
    remove_points.append(row["landmarks"])

remove = pd.DataFrame({"image_id": remove_id, "landmarks": remove_points})

head = {"a": "image_id", "b": "landmarks"}
head = pd.DataFrame(head, index=[0])
head.to_csv("./xxx.csv", mode="w", index=False, header=False)
remove.to_csv("./xxx.csv", mode="a", index=False, header=False)
