import des_temp


def encryption(plaintext, key):
    """
    Performs encryption operation
    :param plaintext: array of 1/0's or booleans
    :param key: array of keys
    :return:
    """
    key1 = key[:64]
    key2 = key[64:128]
    key3 = key[128:]
    return des_temp.encryption(des_temp.decryption(des_temp.encryption(plaintext, key1), key2), key3)


def decryption(ciphertext, key):
    """
    Performs decryption operation
    :param ciphertext: array of 1/0's or booleans
    :param key: array of keys
    :return:
    """
    key1 = key[:64]
    key2 = key[64:128]
    key3 = key[128:]
    return des_temp.decryption(des_temp.encryption(des_temp.decryption(ciphertext, key3), key2), key1)


def encryption_multiple(plaintext, key):
    """
    Performs encryption operation
    :param plaintext: array of 1/0's or booleans
    :param key: array of keys
    :return:
    """
    key1 = key[:, :64]
    key2 = key[:, 64:128]
    key3 = key[:, 128:]
    return des_temp.encryption_multiple(des_temp.decryption_multiple(des_temp.encryption_multiple(plaintext, key1), key2), key3)


def decryption_multiple(ciphertext, key):
    """
    Performs decryption operation
    :param ciphertext: array of 1/0's or booleans
    :param key: array of keys
    :return:
    """
    key1 = key[:, :64]
    key2 = key[:, 64:128]
    key3 = key[:, 128:]
    return des_temp.decryption_multiple(des_temp.encryption_multiple(des_temp.decryption_multiple(ciphertext, key3), key2), key1)


def get_encryption_intermediates(plaintexts, keys):
    """
    Retrieves the intermediate values generated during decryption.
    :param plaintexts: array of 1/0's or booleans
    :param keys: array of keys
    :return:
    """
    key1 = keys[:, :64]
    key2 = keys[:, 64:128]
    key3 = keys[:, 128:]
    intermediates = {"plaintext": plaintexts}
    prefixes = ["encryption1", "decryption2", "encryption3"]

    intermediates_temp = [des_temp.get_encryption_intermediates(plaintexts, key1)]
    intermediates_temp.append(des_temp.get_decryption_intermediates(intermediates_temp[0].get("ciphertext"), key2))
    intermediates_temp.append(des_temp.get_encryption_intermediates(intermediates_temp[1].get("plaintext"), key3))

    for i in range(3):
        for j in intermediates_temp[i].keys():
            intermediates[prefixes[i] + j] = intermediates_temp[i].get(j)

    intermediates["ciphertext"] = intermediates_temp[2].get("ciphertext")
    return intermediates


def get_decryption_intermediates(ciphertexts, keys):
    """
    Retrieves the intermediate values generated during decryption.
    :param ciphertexts: array of 1/0's or booleans
    :param keys: array of keys
    :return:
    """
    key1 = keys[:, :64]
    key2 = keys[:, 64:128]
    key3 = keys[:, 128:]
    intermediates = {"ciphertext": ciphertexts}
    prefixes = ["decryption1", "encryption2", "decryption3"]

    intermediates_temp = [des_temp.get_decryption_intermediates(ciphertexts, key3)]
    intermediates_temp.append(des_temp.get_encryption_intermediates(intermediates_temp[0].get("plaintext"), key2))
    intermediates_temp.append(des_temp.get_decryption_intermediates(intermediates_temp[1].get("ciphertext"), key1))

    for i in range(3):
        for j in intermediates_temp[i].keys():
            intermediates[prefixes[i] + j] = intermediates_temp[i].get(j)

    intermediates["plaintext"] = intermediates_temp[2].get("plaintext")
    return intermediates


def encryption_intermediates_labels():
    labels = ["plaintext", "ciphertext"]
    prefixes = ["encryption1", "decryption2", "encryption3"]

    for label in des_temp.encryption_intermediates_labels():
        labels.append(prefixes[0] + label)
        labels.append(prefixes[2] + label)

    for label in des_temp.decryption_intermediates_labels():
        labels.append(prefixes[1] + label)

    return labels


def decryption_intermediates_labels():
    labels = ["plaintext", "ciphertext"]
    prefixes = ["decryption1", "encryption2", "decryption3"]
    for label in des_temp.decryption_intermediates_labels():
        labels.append(prefixes[0] + label)
        labels.append(prefixes[2] + label)

    for label in des_temp.encryption_intermediates_labels():
        labels.append(prefixes[1] + label)

    return labels
