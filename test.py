import json

class A:
    with open("course.json", encoding="utf-8") as file:
        data = json.load(file)

a = [1, 2, 3, 4, 5]

print([x["Course_name"] for x in A.data])