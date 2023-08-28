import copy

def dict_merger(dict1, dict2):
    result = copy.deepcopy(dict1)
    for key, value in dict2.items():
        if key in dict1:
            result[key] = dict1[key] + value
        else:
            result[key] = value
    return result

def dict_over_freq_filter(test, over_freq=5):
    test_ = dict()
    for keyw, freq in test.items():
        if freq > over_freq:
            test_[keyw] = freq
    return test_
