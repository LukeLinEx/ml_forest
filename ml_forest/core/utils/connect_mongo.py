import json
from pymongo import MongoClient
from os.path import expanduser

path = expanduser("~/.credentials/ml_mongo.json")
f = open(path, 'r')
mongo_connect = f.read()
f.close()
mongo_connect = json.loads(mongo_connect)


def get_credentials(host, database):
    ip = mongo_connect[host]["ip"]
    port = mongo_connect[host]["port"]
    user = mongo_connect["db"][database]["user"]
    pwd = mongo_connect["db"][database]["pwd"]
    client = MongoClient(ip, port)

    return client, user, pwd


def connect_collection(host, database, collection):
    client, user, pwd = get_credentials(host, database)

    db = client[database]
    db.authenticate(user, pwd)
    collection = eval("db." + collection)

    return collection


if __name__ == "__main__":
    f_collection = connect_collection("mltests3mongo", "housing_price", "feature")
    print(f_collection.find_one()["_id"])
