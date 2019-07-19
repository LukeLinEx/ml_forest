import os
from ml_forest.core.constructions.db_handler import *
from ml_forest.core.constructions.io_handler import IOHandler

ih = IOHandler()


def collect_dict_of_obj_2b_rm(c_id, fpaths, _db):
    """

    :param c_id:
    :param fpaths:
    :param _db:
    :return: {
                'CoreInit': 'lst_obj_id',
                'Frame': 'lst_obj_id',
                'Feature': 'lst_obj_id',
                'FTransform': 'lst_obj_id',
                'Label': 'lst_obj_id',
                'LTransform': 'lst_obj_id',
                'GridRecord': 'lst_obj_id',
                'PipeTestData': 'lst_obj_id',
                'TestFeature': 'lst_obj_id'}
    """
    host = _db["host"]
    proj = _db["project"]

    dict_obj_2b_rm = {"CoreInit": [c_id]}

    core = ih.load_obj_from_file(c_id, "CoreInit", fpaths)
    frame_id = core.frame
    dict_obj_2b_rm["Frame"] = [frame_id]

    feature_collection = connect_collection(host=host, database=proj, collection="Feature")
    lst_f = [doc for doc in feature_collection.find({"essentials.frame": frame_id})]
    dict_obj_2b_rm["Feature"] = [doc["_id"] for doc in lst_f]

    lst_ft = [doc["essentials"]["f_transform"] for doc in lst_f]
    lst_ft = [ft_id for ft_id in lst_ft if ft_id is not None]
    dict_obj_2b_rm["FTransform"] = lst_ft

    label_collection = connect_collection(host=host, database=proj, collection="Label")
    lst_l = [doc for doc in label_collection.find({"essentials.frame": frame_id})]
    dict_obj_2b_rm["Label"] = [doc["_id"] for doc in lst_l]

    lst_lt = [doc["essentials"]["l_transform"] for doc in lst_l]
    lst_lt = [lt_id for lt_id in lst_lt if lt_id is not None]
    dict_obj_2b_rm["LTransform"] = lst_lt

    scheme_collection = connect_collection(host=host, database=proj, collection="GridRecord")
    lst_scheme = scheme_collection.find({"essentials.pipe_init": c_id})
    lst_scheme = [doc["_id"] for doc in lst_scheme]
    dict_obj_2b_rm["GridRecord"] = lst_scheme

    pipe_test_data_collection = connect_collection(host=host, database=proj, collection="PipeTestData")
    lst_pipe_test = pipe_test_data_collection.find({"essentials.core_init": c_id})
    lst_pipe_test = [doc["_id"] for doc in lst_pipe_test]
    dict_obj_2b_rm["PipeTestData"] = lst_pipe_test

    test_feature_collection = connect_collection(host=host, database=proj, collection="TestFeature")
    lst_test_f = test_feature_collection.find({
        "essentials.pipe_test": {"$in": dict_obj_2b_rm["PipeTestData"]}
    })
    lst_test_f = [doc["_id"] for doc in lst_test_f]
    dict_obj_2b_rm["TestFeature"] = lst_test_f

    return dict_obj_2b_rm


def get_local_fname(filepath, element, obj_id):
    path = "/".join([filepath["home"], filepath["project"], element])
    fname = path + "/" + str(obj_id) + ".pkl"

    return fname


def deleting_local_file_by_lst_id(dict_obj_2b_rm, fpaths):
    for key in dict_obj_2b_rm:
        if dict_obj_2b_rm[key]:
            print("removing objects from: {}".format(key))
            cmd = "rm " + " ".join(
                [get_local_fname(fpaths[0], key, obj_id) for obj_id in dict_obj_2b_rm[key]]
            )
            os.system(cmd)
            print("\n")


def deleting_docs_from_db(dict_obj_2b_rm, db_):
    host = db_["host"]
    proj = db_["project"]

    for key in dict_obj_2b_rm:
        target_collection = connect_collection(host=host, database=proj, collection=key)
        target_collection.delete_many({"_id": {"$in": dict_obj_2b_rm[key]}})


def rm_obj_by_coreid(c_id, fpaths, db_):
    dict_obj_2b_rm = collect_dict_of_obj_2b_rm(c_id, fpaths, db_)
    deleting_local_file_by_lst_id(dict_obj_2b_rm, fpaths)
    deleting_docs_from_db(dict_obj_2b_rm, db_)


if __name__ == "__main__":
    import os
    import sys
    from bson.objectid import ObjectId

    db_host = "localhost"
    forest_path = os.environ["FORESTPATH"]
    home_path = "{}/local_storage".format(forest_path)
    project = "oversimplified"

    db = {"host": db_host, "project": project}
    filepaths = [{"home": home_path, "project": project}]
    
    core_id = sys.argv[1]
    core_id = ObjectId(core_id)
    rm_obj_by_coreid(core_id, filepaths, db)

