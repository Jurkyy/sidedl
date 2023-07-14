import numpy as np

pc1 = np.array([56, 48, 40, 32, 24, 16, 8,
                0, 57, 49, 41, 33, 25, 17,
                9, 1, 58, 50, 42, 34, 26,
                18, 10, 2, 59, 51, 43, 35,
                62, 54, 46, 38, 30, 22, 14,
                6, 61, 53, 45, 37, 29, 21,
                13, 5, 60, 52, 44, 36, 28,
                20, 12, 4, 27, 19, 11, 3
])

# number left rotations of pc1
left_rotations = np.array([
    1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
])

# permuted choice key (table 2)
pc2 = np.array([
      13, 16, 10, 23, 0, 4,
      2, 27, 14, 5, 20, 9,
      22, 18, 11, 3, 25, 7,
      15, 6, 26, 19, 12, 1,
      40, 51, 30, 36, 46, 54,
      29, 39, 50, 44, 32, 47,
      43, 48, 38, 55, 33, 52,
      45, 41, 49, 35, 28, 31
])

# initial permutation IP
ip = np.array([57, 49, 41, 33, 25, 17, 9, 1,
               59, 51, 43, 35, 27, 19, 11, 3,
               61, 53, 45, 37, 29, 21, 13, 5,
               63, 55, 47, 39, 31, 23, 15, 7,
               56, 48, 40, 32, 24, 16, 8, 0,
               58, 50, 42, 34, 26, 18, 10, 2,
               60, 52, 44, 36, 28, 20, 12, 4,
               62, 54, 46, 38, 30, 22, 14, 6
])

expansion_table = np.array([
                  31, 0, 1, 2, 3, 4,
                  3, 4, 5, 6, 7, 8,
                  7, 8, 9, 10, 11, 12,
                  11, 12, 13, 14, 15, 16,
                  15, 16, 17, 18, 19, 20,
                  19, 20, 21, 22, 23, 24,
                  23, 24, 25, 26, 27, 28,
                  27, 28, 29, 30, 31, 0
])

sbox = np.array([
    # S1
    [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
     0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
     4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
     15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],

    # S2
    [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
     3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
     0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
     13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],

    # S3
    [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
     13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
     13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
     1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],

    # S4
    [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
     13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
     10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
     3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],

    # S5
    [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
     14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
     4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
     11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],

    # S6
    [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
     10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
     9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
     4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],

    # S7
    [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
     13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
     1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
     6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],

    # S8
    [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
     1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
     7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
     2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
])

permut_sbox = np.array([
              15, 6, 19, 20, 28, 11,
              27, 16, 0, 14, 22, 25,
              4, 17, 30, 9, 1, 7,
              23, 13, 31, 26, 2, 8,
              18, 12, 29, 5, 21, 10,
              3, 24
])

# final permutation IP^-1
fp = np.array([
     39, 7, 47, 15, 55, 23, 63, 31,
     38, 6, 46, 14, 54, 22, 62, 30,
     37, 5, 45, 13, 53, 21, 61, 29,
     36, 4, 44, 12, 52, 20, 60, 28,
     35, 3, 43, 11, 51, 19, 59, 27,
     34, 2, 42, 10, 50, 18, 58, 26,
     33, 1, 41, 9, 49, 17, 57, 25,
     32, 0, 40, 8, 48, 16, 56, 24
])


def expand_key(key):
    """"
    Expands key according to nist standard
    """
    round_keys = np.zeros((16, 48), dtype=np.bool_)
    permuted_key = key[pc1]
    c = permuted_key[:28]
    d = permuted_key[28:]
    for i in range(16):
        c = np.roll(c, -left_rotations[i])
        d = np.roll(d, -left_rotations[i])
        round_keys[i] = np.append(c, d)[pc2]
    return round_keys


def expand_key_multiple(key):
    """"
    Expands key according to nist standard
    """
    round_keys = np.zeros((key.shape[0], 16, 48), dtype=np.bool_)
    permuted_key = key[:, pc1]
    c = permuted_key[:, :28]
    d = permuted_key[:, 28:]
    for i in range(16):
        c = np.roll(c, -left_rotations[i], axis=1)
        d = np.roll(d, -left_rotations[i], axis=1)
        round_keys[:, i] = np.append(c, d, axis=1)[:, pc2]
    return round_keys


def encryption(plaintext, key):
    """
    Performs encryption operation
    :param plaintext: array of 1/0's or booleans
    :param key: array of key
    :return:
    """
    round_keys = expand_key(key)
    permuted_plaintext = np.array(plaintext[ip], dtype=np.bool_)
    left_register = permuted_plaintext[:32]
    right_register = permuted_plaintext[32:]
    for i in range(16):
        expanded_r = right_register[expansion_table]
        to_be_subbed = np.logical_xor(round_keys[i], expanded_r)
        output_sboxes = []
        for j in range(8):
            # We need to pad otherwise the 0's are in
            # I know this is a bit of a disaster, sorry
            sub_val = np.packbits(to_be_subbed[j*6+1:j*6+5])[0]//16
            if to_be_subbed[j*6]:
                sub_val += 32
            if to_be_subbed[j*6 + 5]:
                sub_val += 16
            output_val = sbox[j][sub_val]
            output_sboxes = np.append(output_sboxes, np.unpackbits(np.array(output_val, dtype=np.ubyte))[4:])
        output_sboxes = np.array(output_sboxes, dtype=np.bool_)[permut_sbox]
        temp = right_register.copy()
        right_register = np.logical_xor(left_register, output_sboxes)
        left_register = temp
    # The order of L and R are swapped to form the pre-output block.
    output = np.append(right_register, left_register)[fp]
    return output


def decryption(ciphertext, key):
    """
    Performs encryption operation
    :param ciphertext: array of 1/0's or booleans
    :param key: array of key
    :return:
    """
    round_keys = expand_key(key)
    permuted_ciphertext = np.array(ciphertext, dtype=np.bool_)[ip]
    left_register = permuted_ciphertext[32:]
    right_register = permuted_ciphertext[:32]
    for i in range(16):
        expanded_l = left_register[expansion_table]
        to_be_subbed = np.logical_xor(round_keys[15 - i], expanded_l)
        output_sboxes = []
        for j in range(8):
            # We need to pad otherwise the 0's are in
            # I know this is a bit of a disaster, sorry
            sub_val = np.packbits(np.append(np.array([0, 0, 0, 0], dtype=np.bool_),
                                            to_be_subbed[j * 6 + 1:j * 6 + 5]))[0]
            if to_be_subbed[j * 6]:
                sub_val += 32
            if to_be_subbed[j * 6 + 5]:
                sub_val += 16
            output_val = sbox[j][sub_val]

            output_sboxes = np.append(output_sboxes, np.unpackbits(np.array(output_val, dtype=np.ubyte))[4:])
        output_sboxes = np.array(output_sboxes, dtype=np.bool_)[permut_sbox]
        temp = left_register.copy()
        left_register = np.logical_xor(right_register, output_sboxes)
        right_register = temp
    output = np.append(left_register, right_register)[fp]
    return output


def encryption_multiple(plaintexts, keys):
    """
    Performs encryption operation
    :param plaintexts: array of 1/0's or booleans
    :param keys: array of keys
    :return:
    """
    round_keys = expand_key_multiple(keys)
    permuted_plaintext = np.array(plaintexts[:, ip], dtype=np.bool_)
    left_register = permuted_plaintext[:, :32]
    right_register = permuted_plaintext[:, 32:]
    for i in range(16):
        expanded_r = right_register[:, expansion_table]
        to_be_subbed = np.logical_xor(round_keys[:, i], expanded_r)
        output_sboxes = []
        for j in range(8):
            sub_val = np.packbits(to_be_subbed[:, j*6+1:j*6+5], axis=1)//16

            sub_val = sub_val + np.array(to_be_subbed[:, j*6] * 32, dtype=np.uint8).reshape(-1,1)
            sub_val = sub_val + np.array(to_be_subbed[:, j*6+5] * 16, dtype=np.uint8).reshape(-1,1)

            output_val = sbox[j][sub_val]
            if j > 0:

                output_sboxes = np.append(output_sboxes, np.unpackbits(np.array(output_val, dtype=np.ubyte), axis=1)[:, 4:],
                                          axis=1)
            else:
                output_sboxes = np.unpackbits(np.array(output_val, dtype=np.ubyte), axis=1)[:, 4:]
        output_sboxes = np.array(output_sboxes, dtype=np.bool_)[:, permut_sbox]
        temp = right_register.copy()
        right_register = np.logical_xor(left_register, output_sboxes)
        left_register = temp
    # The order of L and R are swapped to form the pre-output block.
    output = np.append(right_register, left_register, axis=1)[:, fp]
    return output


def decryption_multiple(ciphertext, keys):
    """
    Performs encryption operation
    :param ciphertext: array of 1/0's or booleans
    :param keys: array of keys
    :return:
    """
    round_keys = expand_key_multiple(keys)
    permuted_ciphertext = np.array(ciphertext[:, ip], dtype=np.bool_)
    left_register = permuted_ciphertext[:, 32:]
    right_register = permuted_ciphertext[:, :32]
    for i in range(16):
        expanded_l = left_register[:, expansion_table]
        to_be_subbed = np.logical_xor(round_keys[:, 15 - i], expanded_l)
        output_sboxes = []
        for j in range(8):
            # We need to pad otherwise the 0's are in
            # I know this is a bit of a disaster, sorry
            sub_val = np.packbits(to_be_subbed[:, j*6+1:j*6+5], axis=1)//16

            sub_val = sub_val + np.array(to_be_subbed[:, j*6] * 32, dtype=np.uint8).reshape(-1, 1)
            sub_val = sub_val + np.array(to_be_subbed[:, j*6+5] * 16, dtype=np.uint8).reshape(-1, 1)
            output_val = sbox[j][sub_val]

            if j > 0:
                output_sboxes = np.append(output_sboxes,
                                          np.unpackbits(np.array(output_val, dtype=np.ubyte), axis=1)[:, 4:],
                                          axis=1)
            else:
                output_sboxes = np.unpackbits(np.array(output_val, dtype=np.ubyte), axis=1)[:, 4:]
        output_sboxes = np.array(output_sboxes, dtype=np.bool_)[:, permut_sbox]
        temp = left_register.copy()
        left_register = np.logical_xor(right_register, output_sboxes)
        right_register = temp
    output = np.append(left_register, right_register, axis=1)[:, fp]
    return output


def get_encryption_intermediates(plaintexts, keys):
    """
    Retrieves the intermediate values generated during decryption.
    :param plaintexts: array of 1/0's or booleans
    :param keys: array of keys
    :return:
    """
    round_keys = expand_key_multiple(keys)
    intermediates = {"plaintext": plaintexts}
    permuted_plaintext = np.array(plaintexts[:, ip], dtype=np.bool_)
    intermediates[f"permuted_plaintext"] = permuted_plaintext
    left_register = permuted_plaintext[:, :32]
    right_register = permuted_plaintext[:, 32:]
    for i in range(16):
        expanded_r = right_register[:, expansion_table]
        intermediates[f"expanded_R_{i}"] = expanded_r
        to_be_subbed = np.logical_xor(round_keys[:, i], expanded_r)
        intermediates[f"add_round_key_{i}"] = to_be_subbed
        output_sboxes = []
        for j in range(8):
            sub_val = np.packbits(to_be_subbed[:, j*6+1:j*6+5], axis=1)//16

            sub_val = sub_val + np.array(to_be_subbed[:, j*6] * 32, dtype=np.uint8).reshape(-1,1)
            sub_val = sub_val + np.array(to_be_subbed[:, j*6+5] * 16, dtype=np.uint8).reshape(-1,1)

            output_val = sbox[j][sub_val]
            if j > 0:

                output_sboxes = np.append(output_sboxes, np.unpackbits(np.array(output_val, dtype=np.ubyte), axis=1)[:, 4:],
                                          axis=1)
            else:
                output_sboxes = np.unpackbits(np.array(output_val, dtype=np.ubyte), axis=1)[:, 4:]
        intermediates[f"sbox_{i}"] = output_sboxes
        output_sboxes = np.array(output_sboxes, dtype=np.bool_)[:, permut_sbox]
        intermediates[f"permutation_{i}"] = output_sboxes
        temp = right_register.copy()
        right_register = np.logical_xor(left_register, output_sboxes)
        left_register = temp
        intermediates[f"left_register_{i}"] = left_register
        intermediates[f"right_register_{i}"] = right_register
    # The order of L and R are swapped to form the pre-output block.
    output = np.append(right_register, left_register, axis=1)[:, fp]
    intermediates["ciphertext"] = output
    return intermediates


def get_decryption_intermediates(ciphertexts, keys):
    """
    Retrieves the intermediate values generated during decryption.
    :param ciphertexts: array of 1/0's or booleans
    :param keys: array of keys
    :return:
    """
    round_keys = expand_key_multiple(keys)

    intermediates = {"ciphertext": ciphertexts}
    permuted_ciphertext = np.array(ciphertexts[:, ip], dtype=np.bool_)
    intermediates[f"permuted_ciphertext"] = permuted_ciphertext
    left_register = permuted_ciphertext[:, 32:]
    right_register = permuted_ciphertext[:, :32]
    for i in range(16):
        expanded_l = left_register[:, expansion_table]
        intermediates[f"expanded_L_{15-i}"] = expanded_l
        to_be_subbed = np.logical_xor(round_keys[:, 15 - i], expanded_l)
        intermediates[f"add_round_key_{15-i}"] = to_be_subbed
        output_sboxes = []
        for j in range(8):
            # We need to pad otherwise the 0's are in
            # I know this is a bit of a disaster, sorry
            sub_val = np.packbits(to_be_subbed[:, j*6+1:j*6+5], axis=1)//16

            sub_val = sub_val + np.array(to_be_subbed[:, j*6] * 32, dtype=np.uint8).reshape(-1, 1)
            sub_val = sub_val + np.array(to_be_subbed[:, j*6+5] * 16, dtype=np.uint8).reshape(-1, 1)
            output_val = sbox[j][sub_val]

            if j > 0:
                output_sboxes = np.append(output_sboxes,
                                          np.unpackbits(np.array(output_val, dtype=np.ubyte), axis=1)[:, 4:],
                                          axis=1)
            else:
                output_sboxes = np.unpackbits(np.array(output_val, dtype=np.ubyte), axis=1)[:, 4:]
        intermediates[f"sbox_{15-i}"] = output_sboxes
        output_sboxes = np.array(output_sboxes, dtype=np.bool_)[:, permut_sbox]
        intermediates[f"permutation_{15-i}"] = output_sboxes
        output_sboxes = np.array(output_sboxes, dtype=np.bool_)[:, permut_sbox]
        temp = left_register.copy()
        left_register = np.logical_xor(right_register, output_sboxes)
        right_register = temp
        intermediates[f"left_register_{15-i}"] = left_register
        intermediates[f"right_register_{15-i}"] = right_register
    output = np.append(left_register, right_register, axis=1)[:, fp]
    intermediates["plaintext"] = output
    return output


def encryption_intermediates_labels():
    labels = ["plaintext", "permuted_plaintext", "ciphertext"]
    for i in range(16):
        labels.append(f"sbox_{i}")
        labels.append(f"permute_{i}")
        labels.append(f"add_round_key_{i}")
        labels.append(f"expanded_R_{i}")
        labels.append(f"left_register_{i}")
        labels.append(f"right_register_{i}")

    return labels


def decryption_intermediates_labels():
    labels = ["plaintext", "permuted_ciphertext", "ciphertext"]
    for i in range(16):
        labels.append(f"sbox_{i}")
        labels.append(f"permute_{i}")
        labels.append(f"add_round_key_{i}")
        labels.append(f"expanded_L_{i}")
        labels.append(f"left_register_{i}")
        labels.append(f"right_register_{i}")

    return labels
