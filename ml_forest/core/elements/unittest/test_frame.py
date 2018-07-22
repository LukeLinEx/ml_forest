from ml_forest.core.elements.frame_base import Frame


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
    boo = frame.ravel_key((0,1,)) == (15, 30)
    print("test raveling the key (0,1,) OK: {}".format(boo))
    boo = frame.ravel_key((0,2, 1)) == (35, 40)
    print("test raveling the key (0,2,1) OK: {}".format(boo))
    boo = frame.ravel_key((0,1, 2, 1)) == (26, 27)
    print("test raveling the key (0,1,2,1) OK: {}".format(boo))

    print("\n")
    boo = frame.get_single_fold((0,)) == list(range(203))
    print("test creating fold for (0,) OK: {}".format(boo))
    boo = frame.get_single_fold((0,1, 0, 0)) == [75, 76, 77, 78, 79]
    print("test creating fold for (0,1,0,0) OK: {}".format(boo))
    boo = frame.get_single_fold((0,2,2)) == [
        183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202
    ]
    print("test creating fold for (0,2,2) OK: {}".format(boo))

    print("\n")
    boo = frame.get_idx_for_layer(0) == [list(range(203))]
    print("test if get idx for the layer 0 correct: {}".format(boo))
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
    boo = len(frame.get_idx_for_layer(3)) == 45
    print("test if get idx for the layer 3 correct length: {}".format(boo))

    pair = frame.get_train_test_key_pairs(2)
    boo = pair[0] == [
        (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2)
    ]
    print("test if get test_key for the layer 2 correct: {}".format(boo))
    boo = pair[1] == [
        [(0, 0, 1), (0, 0, 2)], [(0, 0, 0), (0, 0, 2)], [(0, 0, 0), (0, 0, 1)],
        [(0, 1, 1), (0, 1, 2)], [(0, 1, 0), (0, 1, 2)], [(0, 1, 0), (0, 1, 1)],
        [(0, 2, 1), (0, 2, 2)], [(0, 2, 0), (0, 2, 2)], [(0, 2, 0), (0, 2, 1)]
    ]
    print("test if get train_key for the layer 2 correct: {}".format(boo))


if __name__ == "__main__":
    test1()
