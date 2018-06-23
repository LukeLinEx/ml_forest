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

    def load_obj_from_file(self, obj, filepaths):
        try:
            obj_id = obj.obj_id
        except AttributeError:
            raise AttributeError("The obj you passed has no obj_id, no way to locate it.")
        filename = str(obj_id) + ".pkl"
        element = obj.decide_element()

        for path in filepaths:
            if "home" in path:
                local_path = "{home}/{project}/{element}/{filename}".format(
                    home=path["home"], project=path["project"], element=element, filename=filename
                )
                return load_from_local(local_path)
            else:
                raise NotImplementedError("We need to implement the io for remote storage")


if __name__ == "__main__":
    path1 = "{a}/{b}/{c}".format(a="abc", b="123", c="luke")
    print(path1)
