import warnings
import numpy as np

from ml_forest.core.elements.frame_base import Frame
from ml_forest.core.elements.label_base import Label
from ml_forest.core.elements.ltrans_base import LTransform


class LFlow(object):
    def label_encoding_transform(self, frame, lab_fed, l_transform):
        """

        :param frame: Frame object
        :param lab_fed: Label object
        :param l_transform: LTransform object
        :return:
        """
        if not isinstance(frame, Frame) or not isinstance(lab_fed, Label) or not isinstance(l_transform, LTransform):
            raise TypeError("Something wrong.")
        new_label_values = l_transform.encode_whole(lab_fed.values)

        return new_label_values, l_transform


class FFlow(object):
    def supervised_fit_transform(self, frame, lst_fed, f_transform, label):
        """

        :param frame:
        :param lst_fed:
        :param f_transform:
        :param label:
        :return:
        """
        l_values, fed_values, prevstage = self.f_collect_components(lst_fed, label)
        work_layer = frame.depth - prevstage

        if work_layer == 0:
            new_feature_values, model_collection, stage = self.last_layer_supervised_train(
                frame, work_layer, fed_values, l_values, f_transform
            )
            # raise NotImplementedError("Not implemented yet. Need to be more careful.")
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

    def unsupervised_fit_transform(self, lst_fed, f_transform):
        l_values, fed_values, prevstage = self.f_collect_components(lst_fed, None)
        new_feature_values, model_collection = self.in_sample_train(fed_values, f_transform)
        stage = prevstage

        # f_transform documenting
        f_transform.record_models(model_collection)

        return new_feature_values, f_transform, stage

    @staticmethod
    def f_collect_components(lst_fed, label):
        if label:
            l_values = label.values
        else:
            l_values = None

        if len(lst_fed) == 1:
            fed_values = lst_fed[0].values
        else:
            fed_values = np.concatenate(list(map(lambda x: x.values, lst_fed)), axis=1)

        prevstage = max(map(lambda x: x.stage, lst_fed))

        return l_values, fed_values, prevstage

    def in_sample_train(self, fed, f_transform):
        model, values = f_transform.fit_whole(fed)
        models = {(0,): model}

        return values, models

    def last_layer_supervised_train(self, frame, work_layer, fed_values, l_values, f_transform):
        model, values = f_transform.fit_singleton(fed_values, l_values, fed_values)
        models = {(0,): model}

        prevstage = frame.depth - work_layer
        stage = prevstage + 1

        return values, models, stage

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
        stage = prevstage + 1  # TODO: should probably change to f_transform.rise

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
