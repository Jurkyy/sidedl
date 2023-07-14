import numpy as np


def ror(x, r):
    """
    Rotates array x to the right by r
    """
    return np.roll(x, r)


def rol(x, r):
    """
    Rotates array x to the left by r
    """
    return np.roll(x, -r)

def ror_multiple(x, r):
    """
    Rotates array x to the right by r
    """
    return np.roll(x, r, axis=1)


def rol_multiple(x, r):
    """
    Rotates array x to the left by r
    """
    return np.roll(x, -r, axis=1)


def boolean_array_to_int(x):
    """
    Helper function that transforms boolean array to an integer
    :param x: the boolean array
    :return: the integer
    """
    result = 0
    for i in range(len(x)):
        if x[-(i+1)]:
            result += pow(2, i)
    return result


def boolean_array_to_int_multiple(x):
    """
    Helper function that transforms boolean array to an integer
    :param x: the boolean array
    :return: the integer
    """
    result = np.zeros(x.shape[0], dtype=np.uint64)
    for i in range(x.shape[1]):
        result = result + np.array(pow(2, i) * x[:, -(i+1)], dtype=np.uint64)
    return result


def int_to_boolean_array(x, length):
    """
    Helper function that transforms an integer to a boolean array of length length
    :param x: integer
    :param length: the length of the array
    :return: the boolean array
    """
    temp = bin(x)
    result = np.zeros(length, dtype=np.bool_)
    for i in range(min(len(temp)-2, length)):
        result[length - i - 1] = temp[len(temp) - i - 1] == '1'

    return result


def int_to_boolean_array_multiple(x, length):
    """
    Helper function that transforms an integer to a boolean array of length length
    :param x: integer
    :param length: the length of the array
    :return: the boolean array
    """
    val = x
    output = np.zeros((x.shape[0], length), dtype=np.bool_)

    for i in range(length):
        output[:, length - i - 1] = val % 2
        val = val // 2
    return output


def boolean_array_add(x, y):
    """
    Addition function for boolean array arithmetic
    :return: the resulting boolean array
    """
    temp = (boolean_array_to_int(x) + boolean_array_to_int(y)) & pow(2, len(x))-1
    return int_to_boolean_array(temp, len(x))


def boolean_array_sub(x, y):
    """
    Substraction function for boolean array arithmetic
    :return: the resulting boolean array
    """

    temp = (boolean_array_to_int(x) - boolean_array_to_int(y)) & pow(2, len(x))-1
    return int_to_boolean_array(temp, len(x))


def boolean_array_add_multiple(x, y):
    """
    Addition function for boolean array arithmetic
    :return: the resulting boolean array
    """
    temp = (boolean_array_to_int_multiple(x) + boolean_array_to_int_multiple(y)) & pow(2, 64)-1
    #print(temp)
    return int_to_boolean_array_multiple(temp, x.shape[1])


def boolean_array_sub_multiple(x, y):
    """
    Substraction function for boolean array arithmetic
    :return: the resulting boolean array
    """

    temp = (boolean_array_to_int_multiple(x) - boolean_array_to_int_multiple(y)) & (pow(2, 64)-1)
    return int_to_boolean_array_multiple(temp, x.shape[1])


def r(x, y, k):
    x = ror(x, 8)
    x = boolean_array_add(x, y)
    x = np.logical_xor(x, k)
    y = rol(y, 3)
    y = np.logical_xor(x, y)
    return x, y


def r_multiple(x, y, k):
    x = ror_multiple(x, 8)
    x = boolean_array_add_multiple(x, y)
    x = np.logical_xor(x, k)
    y = rol_multiple(y, 3)
    y = np.logical_xor(x, y)
    return x, y


def expand_key(key):
    """
    Key expansion for Speck 128
    :param key: base key
    :return:
    """
    key = np.unpackbits(key)
    round_keys = np.zeros((32, 64), np.bool_)
    if len(key) == 192:
        return expand_key192(key)
    if len(key) == 256:
        
        return expand_key256(key)
    a = key[:64]
    b = key[64:]
    round_keys[0] = a
    for i in range(31):

        b, a = r(b, a, int_to_boolean_array(i, len(b)))
        round_keys[i + 1] = a
    return round_keys

def expand_key192(key):
    """
    Key expansion for Speck 128/192
    :param key: base key
    :return:
    """
    a = key[:64]
    b = key[64:128]
    c = key[128:]
    round_keys = np.zeros((33, 64), np.bool_)
    round_keys[0] = a
    i = 1
    while i < 33:
        b, a = r(b, a, int_to_boolean_array(i-1, 64))
        round_keys[i] = a
        i += 1
        c, a = r(c, a, int_to_boolean_array(i-1, 64))
        round_keys[i] = a
        i += 1
    return round_keys


def expand_key256( key):
    """
    Key expansion for Speck 128/256
    :param key: base key
    :return:
    """
    a = key[:64]
    b = key[64:128]
    c = key[128:192]
    d = key[192:]
    round_keys = np.zeros((34, 64), np.bool_)
    round_keys[0] = a
    i = 1
    while i < 34:
        b, a = r(b, a, int_to_boolean_array(i-1, 64))
        round_keys[i] = a
        i += 1
        c, a = r(c, a, int_to_boolean_array(i-1, 64))
        round_keys[i] = a
        i += 1
        d, a = r(d, a, int_to_boolean_array(i-1, 64))
        round_keys[i] = a
        i += 1
    return round_keys


def expand_key_multiple(key):
    """
    Key expansion for Speck 128
    :param key: base key
    :return:
    """
    key = np.unpackbits(key, axis=1)
    round_keys = np.zeros((key.shape[0], 32, 64), np.bool_)
    if key.shape[1] == 192:
        return expand_key192(key)
    if key.shape[1] == 256:
        return expand_key256(key)
    a = key[:, :64]
    b = key[:, 64:]
    round_keys[:, 0] = a
    
    for i in range(31):
        temp = int_to_boolean_array_multiple(np.zeros(key.shape[0], np.uint64) + i, 64)
        b, a = r_multiple(b, a, temp)
        round_keys[:, i + 1] = a
    return round_keys


def expand_key192_multiple(key):
    """
    Key expansion for Speck 128/192
    :param key: base key
    :return:
    """
    a = key[:, :64]
    b = key[:, 64:128]
    c = key[:, 128:]
    round_keys = np.zeros((key.shape[0], 33, 64), np.bool_)
    round_keys[:, 0] = a
    i = 1
    while i < 33:
        b, a = r_multiple(b, a, int_to_boolean_array_multiple(np.zeros(key.shape[0]) + (i-1), 64))
        round_keys[:, i] = a
        i += 1
        c, a = r_multiple(c, a, int_to_boolean_array_multiple(np.zeros(key.shape[0]) + (i-1), 64))
        round_keys[:, i] = a
        i += 1
    return round_keys


def expand_key256_multiple(key):
    """
    Key expansion for Speck 128/256
    :param key: base key
    :return:
    """
    a = key[:, :64]
    b = key[:, 64:128]
    c = key[:, 128:192]
    d = key[:, 192:]
    round_keys = np.zeros((key.shape[0], 34, 64), np.bool_)
    round_keys[:, 0] = a
    i = 1
    while i < 34:
        b, a = r_multiple(b, a, int_to_boolean_array_multiple(np.zeros(key.shape[0]) + (i-1), 64))
        round_keys[:, i] = a
        i += 1
        c, a = r_multiple(c, a, int_to_boolean_array_multiple(np.zeros(key.shape[0]) + (i-1), 64))
        round_keys[:, i] = a
        i += 1
        d, a = r_multiple(d, a, int_to_boolean_array_multiple(np.zeros(key.shape[0]) + (i-1), 64))
        round_keys[:, i] = a
        i += 1
    return round_keys


def encryption(plaintext, key):
    """
    Performs encryption operation
    :param plaintext: array of 1/0's or booleans
    :param key: key
    :return:
    """
    round_keys = expand_key(key)
    temp = np.packbits(round_keys, axis=1)

    x = np.unpackbits(plaintext)[64:]
    y = np.unpackbits(plaintext)[:64]
    x = ror(x, 8)
    x = boolean_array_add(x, y)
    x = np.logical_xor(x, round_keys[0])
    y = rol(y, 3)
    y = np.logical_xor(x, y)

    for i in range(len(round_keys)-1):
        x = ror(x, 8)
        x = boolean_array_add(x, y)
        x = np.logical_xor(x, round_keys[i + 1])
        y = rol(y, 3)
        y = np.logical_xor(x, y)

    return np.packbits(np.append(x, y))


def decryption(ciphertext, key):
    """
    Performs decryption operation
    :param ciphertext: array of 1/0's or booleans
    :param key: key
    :return:
    """
    round_keys = expand_key(key)
    y = np.unpackbits(ciphertext)[64:]
    x = np.unpackbits(ciphertext)[:64]

    y = np.logical_xor(x, y)
    y = ror(y, 3)
    x = np.logical_xor(x, round_keys[len(round_keys)-1])
    x = boolean_array_sub(x, y)
    x = rol(x, 8)
    for i in range(len(round_keys)-1):

        y = np.logical_xor(x, y)
        y = ror(y, 3)
        x = np.logical_xor(x, round_keys[len(round_keys)-2-i])

        x = boolean_array_sub(x, y)
        x = rol(x, 8)

    return np.packbits(np.append(x, y))


def decryption_multiple(ciphertexts, keys):
    """
    Performs decryption operation
    :param ciphertexts: array of 1/0's or booleans
    :param keys: key
    :return:
    """
    round_keys = expand_key_multiple(keys)
    y = np.unpackbits(ciphertexts, axis=1)[:, 64:]
    x = np.unpackbits(ciphertexts, axis=1)[:, :64]

    y = np.logical_xor(x, y)
    y = ror_multiple(y, 3)
    x = np.logical_xor(x, round_keys[:, round_keys.shape[1]-1])
    x = boolean_array_sub_multiple(x, y)
    x = rol_multiple(x, 8)
    for i in range(round_keys.shape[1]-1):
        y = np.logical_xor(x, y)
        y = ror_multiple(y, 3)
        x = np.logical_xor(x, round_keys[:, round_keys.shape[1] - 2 - i])

        x = boolean_array_sub_multiple(x, y)
        x = rol_multiple(x, 8)

    return np.packbits(np.append(y, x, axis=1), axis=1)


def encryption_multiple(plaintext, keys):
    """
    Performs encryption operation
    :param plaintext: array of 1/0's or booleans
    :param keys: key
    :return:
    """
    round_keys = expand_key_multiple(keys)
    x = np.unpackbits(plaintext, axis=1)[:, 64:]
    y = np.unpackbits(plaintext, axis=1)[:, :64]
    x = ror_multiple(x, 8)
    x = boolean_array_add_multiple(x, y)
    x = np.logical_xor(x, round_keys[:, 0])
    y = rol_multiple(y, 3)
    y = np.logical_xor(x, y)

    for i in range(round_keys.shape[1]-1):
        x = ror_multiple(x, 8)
        x = boolean_array_add_multiple(x, y)
        x = np.logical_xor(x, round_keys[:, i + 1])
        y = rol_multiple(y, 3)
        y = np.logical_xor(x, y)

    return np.packbits(np.append(x, y, axis=1), axis=1)


def get_decryption_intermediates(ciphertexts, keys):
    """

    Get intermediate values for decryption
    :param ciphertexts: array of 1/0's or booleans
    :param keys: key
    :return:
    """
    round_keys = expand_key_multiple(keys)
    y = np.unpackbits(ciphertexts, axis=1)[:, 64:]
    x = np.unpackbits(ciphertexts, axis=1)[:, :64]
    intermediates = {"ciphertext": np.append(y, x, axis=1)}
    y = np.logical_xor(x, y)
    intermediates[f"xor_y_{round_keys.shape[1]-1}"] = np.append(y, x, axis=1)
    y = ror_multiple(y, 3)
    intermediates[f"rotate_right_{round_keys.shape[1]-1}"] = np.append(y, x, axis=1)
    x = np.logical_xor(x, round_keys[:, round_keys.shape[1]-1])
    intermediates[f"add_round_key_{round_keys.shape[1]-1}"] = np.append(y, x, axis=1)
    x = boolean_array_sub_multiple(x, y)
    intermediates[f"substract_y_{round_keys.shape[1]-1}"] = np.append(y, x, axis=1)
    x = rol_multiple(x, 8)
    intermediates[f"rotate_left_{round_keys.shape[1]-1}"] = np.append(y, x, axis=1)
    for i in range(round_keys.shape[1]-1):
        y = np.logical_xor(x, y)
        intermediates[f"xor_y_{ round_keys.shape[1] - 2 - i}"] = np.append(y, x, axis=1)
        y = ror_multiple(y, 3)
        intermediates[f"rotate_right_{ round_keys.shape[1] - 2 - i}"] = np.append(y, x, axis=1)
        x = np.logical_xor(x, round_keys[:, round_keys.shape[1] - 2 - i])
        intermediates[f"add_round_key_{ round_keys.shape[1] - 2 - i}"] = np.append(y, x, axis=1)
        x = boolean_array_sub_multiple(x, y)
        intermediates[f"substract_y_{ round_keys.shape[1] - 2 - i}"] = np.append(y, x, axis=1)
        x = rol_multiple(x, 8)
        intermediates[f"rotate_left_{ round_keys.shape[1] - 2 - i}"] = np.append(y, x, axis=1)
    intermediates["plaintext"] = np.append(y, x, axis=1)
    return intermediates


def get_encryption_intermediates(plaintext, keys):
    """
    Get intermediate values for encryption
    :param plaintext: array of 1/0's or booleans
    :param keys: key
    :return:
    """

    round_keys = expand_key_multiple(keys)
    x = np.unpackbits(plaintext, axis=1)[:, 64:]
    y = np.unpackbits(plaintext, axis=1)[:, :64]
    intermediates = {"plaintext": np.append(y, x, axis=1)}
    for i in range(round_keys.shape[1]):
        x = ror_multiple(x, 8)
        intermediates[f"rotate_right_{i}"] = np.append(y, x, axis=1)
        x = boolean_array_add_multiple(x, y)
        intermediates[f"add_y_{i}"]= np.append(y, x, axis=1)
        x = np.logical_xor(x, round_keys[:, i])
        intermediates[f"add_round_key_{i}"] = np.append(y, x, axis=1)
        y = rol_multiple(y, 3)
        intermediates[f"rotate_left_{i}"] = np.append(y, x, axis=1)
        y = np.logical_xor(x, y)
        intermediates[f"xor_y_{i}"] = np.append(y, x, axis=1)

    intermediates["ciphertext"] = np.append(y, x, axis=1)
    return intermediates


def encryption_intermediates_labels(key_bits=128):
    labels = ["plaintext", "ciphertext"]
    num_rounds = {128: 32, 192: 33, 256: 34}
    for i in range(num_rounds.get(key_bits)):
        labels.append(f"rotate_right_{i}")
        labels.append(f"add_y_{i}")
        labels.append(f"add_round_key_{i}")
        labels.append(f"rotate_left_{i}")
        labels.append(f"xor_y_{i}")
    return labels


def decryption_intermediates_labels(key_bits=128):
    labels = ["plaintext", "ciphertext"]
    num_rounds = {128: 32, 192: 33, 256: 34}
    for i in range(num_rounds.get(key_bits)):
        labels.append(f"rotate_right_{i}")
        labels.append(f"substract_y_{i}")
        labels.append(f"add_round_key_{i}")
        labels.append(f"rotate_left_{i}")
        labels.append(f"xor_y_{i}")
    return labels

