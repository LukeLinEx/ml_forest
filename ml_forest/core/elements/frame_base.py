import numpy as np
from ml_forest.core.elements.identity import Base
from ml_forest.core.constructions.db_handler import DbHandler

__author__ = 'LukeLin'


class Frame(Base):
    def __init__(self, num_observations, lst_layers, **kwargs):
        """

        :param num_observations: positive int
        :param lst_layers: list of int
        :param db: list of dictionaries
        :param filepaths: list of dictionaries
        :return:
        """
        self.__check(num_observations, lst_layers)

        super(Frame, self).__init__(**kwargs)
        self.__num_observations = num_observations
        self.__lst_layers = lst_layers
        self.__essentials = {
            'num_observations': self.__num_observations,
            'lst_layers': self.__lst_layers
        }
        self.__depth = len(lst_layers)

        if type(self) == Frame:
            if self.db:
                dh = DbHandler()
                obj_id = dh.init_doc(self)
                self.obj_id = obj_id
            # TODO: saving the obj into file
            if self.filepaths:
                raise NotImplementedError("Implement the saving objects")

    @staticmethod
    def __check(num_observations, lst_layers):
        """

        :type num_observations: int
        """
        if not np.array(lst_layers).dtype == int:
            raise TypeError('The lst_layer should contain only integers.')
        if not isinstance(num_observations, int):
            raise TypeError('the num_observations should be an integer')

        fold_num_deepest_layer = 1
        for i in lst_layers:
            fold_num_deepest_layer *= i

        print('The smallest fold contains %d observations' % (num_observations//fold_num_deepest_layer))

    def create_structure(self, layer=None):
        """

        create the structure for a particular layer:
        if lst_layers = [2, 3, 2], then:
        the structure for the 1st layer is:
        [(0,), (1,)]
        the structure for the 2nd layer is:
        [(0,0),(0,1),(0,2), (1,0),(1,1),(1,2), (2,0),(2,1),(2,2)]
        the structure for the 3rd layer (which is also the structure for the whole frame) is:
        [(0,0, 0),(0,0, 1), (0,1, 0),(0,1, 1), (0,2, 0),(0,2, 1),
         (1,0, 0),(1,0, 1), (1,1, 0),(1,1, 1), (1,2, 0),(1,2, 1),
         (2,0, 0),(2,0, 1), (2,1, 0),(2,1, 1), (2,2, 0),(2,2, 1)]

        :param layer: int
        :param layer: int
        :return:
        """
        structure = []
        lst_layers = self.__lst_layers
        if not layer:
            layer = len(lst_layers)

        for i in range(layer):
            if i == 0:
                structure = map(lambda x: [x], range(lst_layers[i]))
            else:
                tmp = []
                for key in structure:
                    for j in range(lst_layers[i]):
                        tmp.append(key + [j])
                structure = tmp

        return list(map(lambda x: tuple(x), structure))

    def ravel_key(self, key):
        """
        Flatten the hierarchical key to integers that indicate the starting and ending folds (in the lowest layer)

        Ex:

        frame = Frame(20, [2, 5]) # Lowest layer consists of ten folds
        print frame.ravel_key((0,)), '\n' # This indicates the first fold of the first layer,
                                          # which contains 5 layers in the next (in this case, the last) layer
                                          # so the result of ravel is (0, 5)
        print frame.ravel_key((0, 0)), frame.ravel_key((0, 1)), frame.ravel_key((0, 2)),
              frame.ravel_key((0, 3)), frame.ravel_key((0, 4))


        :param key: hierarchical keys which indicate the fold
        :return:
        """
        reverse_lst_layers = self.lst_layers[::-1]
        diff = len(reverse_lst_layers) - len(key)
        key = (list(key) + [0] * diff)

        for i in range(len(key)):
            if not self.lst_layers[i] > key[i]:
                raise ValueError('The key number %d is not valid' % (i + 1))

        reverse_key = key[::-1]
        start = reverse_key[0]
        unit = 1
        for i in range(1, len(reverse_key)):
            unit *= reverse_lst_layers[i - 1]
            start += reverse_key[i] * unit

        adding = 1
        for i in reverse_lst_layers[:diff]:
            adding *= i

        return start, start + adding

    def get_smallest_fold_start(self, flat_key):
        """

        :param flat_key: indicates the location of the fold in the lowest layer.
        :return:
        """
        n = self.num_observations

        fold_num_deepest_layer = 1
        for i in self.lst_layers:
            fold_num_deepest_layer *= i

        quo = n // fold_num_deepest_layer
        rem = n % fold_num_deepest_layer

        if flat_key >= rem:
            start = rem * (quo + 1) + (flat_key - rem) * quo
        else:
            start = flat_key * (quo + 1)

        return start

    def get_single_fold(self, key):
        """
        with the hierarchical key, find the starting idx and the ending idx

        :param key:
        :return:
        """
        start, end = self.ravel_key(key)
        start = self.get_smallest_fold_start(start)
        end = self.get_smallest_fold_start(end)
        return list(range(start, end))

    def get_idx_for_layer(self, layer):
        if not isinstance(layer, int):
            raise TypeError('The height should be an integer.\n')
        if layer < 1:
            raise ValueError('The height should be at least 1.\n')
        if layer > self.depth:
            raise ValueError("The height can't exceed the depth of the Frame.\n")
        lst_idx = self.create_structure(layer)
        return [self.get_single_fold(key) for key in lst_idx]

# TODO: decide if we need __eq__
#     def __eq__(self, other):
#         return self.__num_observations == other.__num_observations and self.__lst_layers == other.__lst_layers

    @property
    def lst_layers(self):
        return self.__lst_layers

    @property
    def num_observations(self):
        return self.__num_observations

    @property
    def depth(self):
        return self.__depth


# TODO: decide a derived class that can be easily used to initialize
# class FrameInitByDeepestLayer(Frame):
#     """
#     The most important difference from Frame class is get_smallest_fold_start
#     """
#
#     def __init__(self, num_observations, lst_layers, len_folds_deepest_layer, **kwargs):
#         super(FrameInitByDeepestLayer, self).__init__(num_observations, lst_layers, **kwargs)
#         self.__len_folds_deepest_layer = len_folds_deepest_layer
#         if type(self) == FrameInitByDeepestLayer:
#             self.save2files()
#
#     def get_smallest_fold_start(self, flat_key):
#         if flat_key == 0:
#             return 0
#         else:
#             start = 0
#             for i in self.len_folds_deepest_layer[:flat_key]:
#                 start += i
#             return start
#
#     @property
#     def len_folds_deepest_layer(self):
#         return self.__len_folds_deepest_layer


def test1():
    print('Creating a frame:')
    frame = Frame(203, [3, 3, 5])

    print("\n")
    print("Test some attributes:")
    boo = frame.lst_layers == [3, 3, 5]
    print("tset lst_layers correct: {}".format(boo))
    boo = frame.num_observations == 203
    print("test num_observations correct: {}".format(boo))
    boo = frame.depth == 3
    print("test depth correct: {}".format(boo))

    print("\n")
    print('Test ravel_key:')
    boo = frame.ravel_key((1,)) == (15, 30)
    print("test raveling the key (1,) OK: {}".format(boo))
    boo = frame.ravel_key((2, 1)) == (35, 40)
    print("test raveling the key (2,1) OK: {}".format(boo))
    boo = frame.ravel_key((1, 2, 1)) == (26, 27)
    print("test raveling the key (1,2,1) OK: {}".format(boo))

    print("\n")
    boo = frame.get_single_fold((1, 0, 0)) == [75, 76, 77, 78, 79]
    print("test creating fold for (1,0,0) OK: {}".format(boo))
    boo = frame.get_single_fold((2,2)) == [
        183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202
    ]
    print("test creating fold for (2,2) OK: {}".format(boo))

    print("\n")
    boo = frame.get_idx_for_layer(1) == [
        list(range(75)),
        list(range(75, 143)),
        list(range(143, 203))
    ]
    print("test if get idx for the layer 1 correct: {}".format(boo))
    boo = frame.get_idx_for_layer(2) == [
        list(range(25)), list(range(25, 50)), list(range(50, 75)), list(range(75, 100)),
        list(range(100, 123)), list(range(123, 143)), list((range(143, 163))),
        list(range(163, 183)), list(range(183, 203))
    ]
    print("test if get idx for the layer 2 correct: {}".format(boo))




if __name__ == '__main__':
    # bucket = "mltests3mongo"
    # project = "housing_price"
    #
    # db = {"host": bucket, "project": project}
    frame = Frame(20, [2, 5])
    print(frame)


    def get_folds(frame, current_layer):
        if current_layer == 1:
            key_target = [tuple()]
        else:
            target_layer = current_layer - 1
            key_target = frame.create_structure(target_layer)

        key_current = frame.create_structure(current_layer)
        hash_current = dict(
            zip(key_current, frame.get_idx_for_layer(current_layer)))  # This should work for both py2 & py3
        return key_target, key_current, hash_current

    def out_sample_train(frame, current_layer):
        # TODO: currently not good for the top layer
        assert current_layer>0, "At this point, no more clean fold possible. Use train_final instead"
        key_target, key_current, hash_current = get_folds(frame, current_layer)

        train_keys = []
        test_keys = []
        for test_k in key_current:
            train_k = list(filter(lambda k: k[0] == test_k[0] and k != test_k, key_current))
            train_keys.append(train_k)
            test_keys.append(test_k)

        return test_keys, train_keys


    # for p in zip(*out_sample_train(frame, 2)):
    #     print(p[0])
    #     print(p[1])
    #     print("\n")
    print(get_folds(frame, 1))



        # for k in key_target:
        #     collect = filter(lambda i: i[:len(k)] == k, key_current) # (0,0), (0,1), (0,2) are under (0,);
        #                                                              # (2,1,0), (2,1,1), (2,1,2) are under (2,1,)
        #
        #     for test_key in collect:
        #         train_keys = [key for key in collect if key !=  test_key]
        #         train_idx = []
        #         for key in train_keys:
        #             train_idx += hash_current[key]
        #         test_idx = hash_current[test_key]


