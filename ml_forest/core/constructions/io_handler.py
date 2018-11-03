import os, subprocess
from bson.objectid import ObjectId

from ml_forest.core.utils.local_file_io import save_local, load_from_local


class IOHandler(object):
    def __init__(self):
        pass

    @staticmethod
    def save_obj2file(obj):
        element = obj.decide_element()
        filename = str(obj.obj_id) + ".pkl"

        for path in obj.filepaths:
            if 'bucket' in path:
                raise NotImplementedError("We need to implement for uploading to S3 bucket")
            elif 'home' in path:
                local_path = "{home}/{project}/{element}/{filename}".format(
                    home=path["home"], project=path["project"], element=element, filename=filename
                )
                save_local(obj, local_path)

    def load_obj_from_file(self, obj_id, element, filepaths, cache=None):
        """

        :param obj_id: ObjectId
        :param element: str
        :param filepaths: dict
        :param cache:
        :return:
        """
        if not isinstance(obj_id, ObjectId):
            raise TypeError("The parameter obj_id should be a bson.objectid.ObjectId")
        filename = str(obj_id) + ".pkl"

        # TODO: Need to find a better way to cache
        if cache is None:
            cache = {}
        if obj_id in cache:
            return cache[obj_id]

        for path in filepaths:
            if "home" in path:
                local_path = "{home}/{project}/{element}/{filename}".format(
                    home=path["home"], project=path["project"], element=element, filename=filename
                )
                return load_from_local(local_path)
            else:
                raise NotImplementedError("We need to implement the io for remote storage")

    def rm_local_objs_of_a_class(self, lst_obj_id, obj_class, filepath):
        dirpath = "/".join(filepath[key] for key in ["home", "project"])
        dirpath = dirpath + "/" + obj_class

        rmlst = {str(oid) + ".pkl" for oid in lst_obj_id}

        lst_existing_fnames = subprocess.run(
            ["ls", dirpath], stdout=subprocess.PIPE
        ).stdout.decode().split("\n")
        lst_existing_fnames = {f for f in lst_existing_fnames}

        real_rmlst = rmlst.intersection(lst_existing_fnames)

        cmd = ""
        if real_rmlst:
            cmd = "rm " + " ".join(real_rmlst)

        os.system(cmd)


if __name__ == "__main__":
    path1 = "{a}/{b}/{c}".format(a="abc", b="123", c="luke")
    print(path1)
