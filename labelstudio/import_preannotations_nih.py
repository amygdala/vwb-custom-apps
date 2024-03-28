# Import tasks with pre-annotations (predictions) using SDK,
# then calculate agreement scores (accuracy) per tasks.

# from evalme.metrics import (
#     get_agreement,
# )  # run first `pip install label-studio-evalme` to use this package
from label_studio_sdk import Client
import pandas as pd

LABEL_STUDIO_URL = "http://localhost:8080"
# API_KEY = "6f6776926635e0d49a23498e8e8e6c6725dcdc46"
API_KEY = "b475694381f81200724794431a491fb12c6c6359"

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)


project = ls.start_project(
    title="SDK proj: NIH single label",
    label_config="""
    <View>
    <Image name="image" value="$image"/>
    <Choices name="image_class" toName="image" choice="multiple">
        <Choice value="Atelectasis"/>
        <Choice value="Consolidation"/>
        <Choice value="Infiltration"/>
        <Choice value="Pneumothorax"/>
        <Choice value="Edema"/>
        <Choice value="Emphysema"/>
        <Choice value="Fibrosis"/>
        <Choice value="Effusion"/>
        <Choice value="Pneumonia"/>
        <Choice value="Pleural_Thickening"/>
        <Choice value="Cardiomegaly"/>
        <Choice value="Nodule"/>
        <Choice value="Mass"/>
        <Choice value="Hernia"/>
        <Choice value="NoFinding"/>
    </Choices>
    </View>
    """,
)



# project.import_tasks(
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
# )


# project.import_tasks(
#     [
#         {
#             "image": f"https://data.heartex.net/open-images/train_0/mini/0045dd96bf73936c.jpg",
#             "pet": "Dog",
#         },
#         {
#             "image": f"https://data.heartex.net/open-images/train_0/mini/0083d02f6ad18b38.jpg",
#             "pet": "Cat",
#         },
#     ],
#     preannotated_from_fields=["pet"],
# )


# pd.read_csv("https://storage.googleapis.com/vwb-solns-public/data/dogs_and_cats/cd.csv")

# project.import_tasks("https://storage.googleapis.com/vwb-solns-public/data/dogs_and_cats/cd.csv", preannotated_from_fields=["category"])
project.import_tasks("./newnih_chest_xrays2000.csv", preannotated_from_fields=["label"])


# tasks_ids = project.get_tasks_ids()
# project.create_prediction(tasks_ids[0], result="Dog", model_version="1")


# predictions = [
#     {"task": tasks_ids[0], "result": "Dog", "score": 0.9},
#     {"task": tasks_ids[1], "result": "Cat", "score": 0.8},
# ]
# project.create_predictions(predictions)

# print("Pre-annotation agreement scores:")

# total_score = 0
# n = 0
# for task in project.tasks:
#     score = get_agreement(task["annotations"][0], task["predictions"][0])
#     print(f'{task["id"]} ==> {score}')
#     total_score += score
#     n += 1

# print(f"Pre-annotation accuracy: {100 * total_score / n: .0f}%")
