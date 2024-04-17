import json

# file = "data/ls/cd.csv"
# file = "newnih_chest_xrays_tiny.csv"
file = "newnih_chest_xrays2000.csv"


jsonl = []

with open(file, "r") as f:
    with open(f"{file}.json", "w") as nf:
        line = f.readline() #discard header. (TODO: generalize by getting header labels)
        while True:
            line = f.readline()
            if not line:
                break
            elts = line.split(",")
            if len(elts) < 2:
                break
            image = elts[0].strip()
            category = elts[1].strip()
            data = {"image": f"{image}"}
            value = {"choices": [f"{category}"]}
            result = [
                {
                    "from_name": "image_class",
                    "to_name": "image",
                    "type": "choices",
                    "value": value,
                }
            ]
            predictions = [{"result": result, "score": 1}]
            json_elt = {"data": data, "predictions": predictions}
            print(json.dumps(json_elt))
            jsonl.append(json_elt)
        nf.write(json.dumps(jsonl, indent=2))


#     [
#         {
#             "data": {
#                 "image": "https://data.heartex.net/open-images/train_0/mini/0045dd96bf73936c.jpg"
#             },
#             "predictions": [
#                 {
#                     "result": [
#                         {
#                             "from_name": "image_class",
#                             "to_name": "image",
#                             "type": "choices",
#                             "value": {"choices": ["Dog"]},
#                         }
#                     ],
#                     "score": 0.87,
#                 }
#             ],
#         },
#         {
#             "data": {
#                 "image": "https://data.heartex.net/open-images/train_0/mini/0083d02f6ad18b38.jpg"
#             },
#             "predictions": [
#                 {
#                     "result": [
#                         {
#                             "from_name": "image_class",
#                             "to_name": "image",
#                             "type": "choices",
#                             "value": {"choices": ["Cat"]},
#                         }
#                     ],
#                     "score": 0.65,
#                 }
#             ],
#         },
#     ]
