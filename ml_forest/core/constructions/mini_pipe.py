import numpy as np

from ml_forest.core.elements.feature_base import Feature
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.stacking_node import FNode, LNode


class MiniPipe(object):
    def flow_to(self, node):
        db = node.pipe_init.db
        filepaths = node.pipe_init.filepaths

        if isinstance(node, FNode):
            if node.f_transform.rise == 1:
                feature, f_transform = self.supervised_fit_transform(node)
                node.obj_id = feature.obj_id

                feature.set_db(db)
                feature.set_filepaths(filepaths)
                feature.save_db_file()

                f_transform.set_db(db)
                f_transform.set_filepaths(filepaths)
                f_transform.save_db_file()

                del feature
                del f_transform

        # TODO: Next
        elif isinstance(node, LNode):
            raise NotImplementedError("Do It !!!!!")
        ########################################

        node.obj_id = 123

    @staticmethod
    def f_collect_components(f_node):
        ih = IOHandler()
        frame = f_node.pipe_init.frame
        frame = ih.load_obj_from_file(
            obj_id=frame, element="Frame", filepaths=f_node.pipe_init.filepaths
        )

        l_node = f_node.l_node
        label = ih.load_obj_from_file(
            obj_id=l_node.obj_id, element="Label", filepaths=f_node.pipe_init.filepaths
        )
        l_values = label.values

        lst_fed = []
        for f in f_node.lst_fed:
            f_id = f.obj_id
            fed = ih.load_obj_from_file(obj_id=f_id, element="Feature", filepaths=f_node.pipe_init.filepaths)
            lst_fed.append(fed)
        if len(lst_fed) == 1:
            fed_values = lst_fed[0].values
        else:
            fed_values = np.concatenate(map(lambda x: x.values, lst_fed), axis=1)

        prevstage = max(map(lambda x: x.stage, lst_fed))
        work_layer = frame.depth - prevstage

        return frame, l_values, fed_values, work_layer

    @staticmethod
    def out_sample_train(frame, work_layer, fed_values, l_values, f_transform):
        lst_test_keys, lst_train_keys = frame.get_train_test_key_pairs(work_layer)

        values = []
        models = []
        for i in range(len(lst_test_keys)):
            test_key = lst_test_keys[i]
            test_idx = frame.get_single_fold(test_key)

            train_key_pack = lst_train_keys[i]
            train_idx = []
            for key in train_key_pack:
                train_idx.extend(frame.get_single_fold(key))

            x_train = fed_values[train_idx, :]
            y_train = l_values[train_idx, :]
            x_test = fed_values[test_idx, :]

            model, tmp = f_transform.fit_singleton(x_train, y_train, x_test)
            models.append((test_key, model))
            if len(tmp.shape) == 1:
                tmp = tmp.reshape((-1, 1))
            values.append(tmp)

        values = np.concatenate(values, axis=0)

        return values, dict(models)

    @staticmethod
    def out_sample_train_with_tuning(frame, work_layer, fed_values, l_values, f_transform):
        lst_test_keys, lst_train_keys = frame.get_train_test_key_pairs(work_layer)
        if min(map(len, lst_train_keys)) < 2:
            raise ValueError("Training portion has less than 2 folds, can't train with validation.")

        values = []
        models = []
        for i in range(len(lst_test_keys)):
            test_key = lst_test_keys[i]
            test_idx = frame.get_single_fold(test_key)

            train_key_pack = lst_train_keys[i]
            validation_key = train_key_pack[-1]
            validation_idx = frame.get_single_fold(validation_key)

            train_key_pack = train_key_pack[:-1]
            train_idx = []
            for key in train_key_pack:
                train_idx.extend(frame.get_single_fold(key))

            x_train = fed_values[train_idx, :]
            y_train = l_values[train_idx, :]
            x_validation = fed_values[validation_idx, :]
            y_validation = fed_values[validation_idx, :]
            x_test = fed_values[test_idx, :]

            model, tmp = f_transform.fit_singleton(x_train, y_train, x_validation, y_validation, x_test)
            models.append((test_key, model))
            if len(tmp.shape) == 1:
                tmp = tmp.reshape((-1, 1))
            values.append(tmp)

        values = np.concatenate(values, axis=0)

        return values, dict(models)

    def supervised_fit_transform(self, f_node):
        f_transform = f_node.f_transform
        frame, l_values, fed_values, work_layer = self.f_collect_components(f_node)

        if work_layer == 0:
            raise NotImplementedError("Not implemented yet. Need to be more careful.")
        else:
            if f_transform.tuning:
                new_feature_values, model_collection = self.out_sample_train_with_tuning(
                    frame, work_layer, fed_values, l_values, f_transform
                )
            else:
                new_feature_values, model_collection = self.out_sample_train(
                    frame, work_layer, fed_values, l_values, f_transform
                )

            # f_transform documenting + saving
            f_transform.record_models(model_collection)

            # feature documenting + saving
            feature = Feature(
                frame=f_node.pipe_init.frame,  # get the obj_id of frame
                lst_fed=[f.obj_id for f in f_node.lst_fed], # get the obj_id from lst of fnodes
                f_transform=f_transform.obj_id, label=f_node.l_node.obj_id,
                values=new_feature_values
            )

        return feature, f_transform
