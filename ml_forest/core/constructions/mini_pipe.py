import numpy as np

from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.stacking_node import FNode, LNode


class MiniPipe(object):
    def flow_to(self, node):
        if isinstance(node, FNode):
            if node.f_transform.rise == 1:
                fh = FHandler()
                f_values, f_transform, stage = fh.supervised_fit_transform(node)
                return f_values, f_transform, stage
            else:
                fh = FHandler()
                f_values, f_transform = fh.unsupervised_fit_transform(node)
                return f_values, f_transform

        elif isinstance(node, LNode):
            lh = LHandler()
            l_values, l_transform = lh.label_encoding_transform(node)
            return l_values, l_transform


class LHandler(object):
    def label_encoding_transform(self, l_node):
        if not isinstance(l_node, LNode):
            raise TypeError("LHandler can only handle a LNode")

        l_transform = l_node.l_transform
        frame, lab_fed = self.l_collect_components(l_node)
        new_label_values = l_transform.encode_whole(lab_fed)

        return new_label_values, l_transform

    @staticmethod
    def l_collect_components(l_node):
        ih = IOHandler()
        frame = ih.load_obj_from_file(
            obj_id=l_node.pipe_init.frame, element="Frame", filepaths=l_node.pipe_init.filepaths
        )

        lab_fed = ih.load_obj_from_file(
            obj_id=l_node.lab_fed.obj_id, element="Label", filepaths=l_node.pipe_init.filepaths
        )
        lab_fed = lab_fed.values

        return frame, lab_fed


class FHandler(object):
    def supervised_fit_transform(self, f_node):
        f_transform = f_node.f_transform
        frame, l_values, fed_values, prevstage = self.f_collect_components(f_node)
        work_layer = frame.depth - prevstage

        if work_layer == 0:
            raise NotImplementedError("Not implemented yet. Need to be more careful.")
        else:
            if f_transform.tuning:
                new_feature_values, model_collection, stage = self.out_sample_train_with_tuning(
                    frame, work_layer, fed_values, l_values, f_transform
                )
            else:
                new_feature_values, model_collection, stage = self.out_sample_train(
                    frame, work_layer, fed_values, l_values, f_transform
                )

            # f_transform documenting
            f_transform.record_models(model_collection)

        return new_feature_values, f_transform, stage

    def unsupervised_fit_transform(self, f_node):
        raise NotImplementedError("Need to implement for unsupervised learning")

    @staticmethod
    def f_collect_components(f_node):
        ih = IOHandler()
        frame = ih.load_obj_from_file(
            obj_id=f_node.pipe_init.frame, element="Frame", filepaths=f_node.pipe_init.filepaths
        )

        label = ih.load_obj_from_file(
            obj_id=f_node.l_node.obj_id, element="Label", filepaths=f_node.pipe_init.filepaths
        )
        l_values = label.values

        lst_fed = []
        for f in f_node.lst_fed:
            fed = ih.load_obj_from_file(obj_id=f.obj_id, element="Feature", filepaths=f_node.pipe_init.filepaths)
            lst_fed.append(fed)
        if len(lst_fed) == 1:
            fed_values = lst_fed[0].values
        else:
            fed_values = np.concatenate(list(map(lambda x: x.values, lst_fed)), axis=1)

        prevstage = max(map(lambda x: x.stage, lst_fed))

        return frame, l_values, fed_values, prevstage

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
        prevstage = frame.depth - work_layer
        stage = prevstage + 1

        return values, dict(models), stage

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
            y_validation = l_values[validation_idx, :]
            x_test = fed_values[test_idx, :]

            model, tmp = f_transform.fit_singleton(x_train, y_train, x_validation, y_validation, x_test)
            models.append((test_key, model))
            if len(tmp.shape) == 1:
                tmp = tmp.reshape((-1, 1))
            values.append(tmp)

        values = np.concatenate(values, axis=0)
        prevstage = frame.depth - work_layer
        stage = prevstage + 1

        return values, dict(models), stage
