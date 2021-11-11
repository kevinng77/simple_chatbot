import json


with open('attraction_db.json','r')as fp:
    file = json.load(fp)


place = [x[0] for x in file]
print(place)
with open('place.txt','w')as fp:
    for i in place:
        fp.write(f"{i}\n")

