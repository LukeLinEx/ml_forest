import pickle


def save_local(obj, local_path):
    try:
        with open(local_path, 'wb') as fp:
            pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # TODO: save log file
        element = obj.decide_element()
        _id = obj.obj_id
        msg = "The {} with obj_id {} is not saved".format(element, _id)
        raise IOError(msg)


def load_from_local(local_path):
    with open(local_path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj
