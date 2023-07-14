"""
    Class that describes the modifying and using of files with the .h5 extension.
"""

import h5py
import numpy as np


def get_h5_list_of_group_names(file_path):
    """
        Return group names from a h5 file.

        From https://docs.h5py.org/en/stable/high/group.html documentation, list of groups in a h5py file are obtained with .keys() method

        :param file_path: filepath for the h5 file.
        :return
            - list with group names
    """

    group_names = []
    dataset_names = []

    def recursively_visit_groups_and_subgroups(name, node):
        if isinstance(node, h5py.Dataset):
            dataset_names.append(name)
        else:
            group_names.append(name)

    with h5py.File(file_path, 'r') as f:
        f.visititems(recursively_visit_groups_and_subgroups)

    return group_names, dataset_names


def get_dataset_fields(file_path, dataset_name):
    """
       Return list of dataset field names

       :param file_path: filepath for the h5 file.
       :param dataset_name: dataset name
       :return
           - dictionary with list of value fields
    """

    dataset_fields = []

    in_file = h5py.File(file_path, "r")
    fields = in_file[dataset_name].dtype.fields
    if fields is not None:
        for field in fields:
            dataset_fields.append(field)
    return dataset_fields


def get_h5_dataset_values(file_path, dataset_name):
    """
        Return values for the correspondent dataset.

        :param file_path: filepath for the h5 file.
        :param dataset_name: dataset_name
        :return
            - return values for the correspondent dataset
    """

    in_file = h5py.File(file_path, "r")
    dataset = in_file[dataset_name]
    return np.array(dataset, dtype=dataset.dtype)


def get_h5_dataset_field_values(file_path, dataset_name, dataset_field):
    """
        Return values for the correspondent dataset field.

        :param file_path: filepath for the h5 file.
        :param dataset_name: dataset_name
        :param dataset_field: dataset_field
        :return
            - return values for the correspondent dataset field
    """

    in_file = h5py.File(file_path, "r")
    return in_file[dataset_name][dataset_field]


def read_h5(file_path, num_profiling_traces, num_attack_traces, first_feature, num_features):
    """
    Read function for files with the .h5 extension.

    Takes a set of parameters to read out a .h5 file in the desired way. Returns a dict containing all relevant
    information.

    :param file_path            : the path of the .h5 file to be read.
    :param num_profiling_traces : the amount of profiling traces to use.
    :param num_attack_traces    : the amount of attack traces to use.
    :param first_feature        : the first feature to use for the designated range.
    :param num_features         : the total amount of features to use.

    :return:
        A python dictionary containing the following:

        read_dataset_dict = {
            "profiling_traces"  : the specified amount of profiling traces, X and Y split in a tuple.
            "validation_traces" : the specified amount of validation traces, X and Y split in a tuple.
            "attack_traces"     : the specified amount of attack traces, X and Y split in a tuple.
            "plaintexts"        : a 3-tuple containing the profiling, validation and attack plaintexts in that order.
            "ciphertexts"       : a 3-tuple containing the profiling, validation and attack ciphertexts in that order.
            "keys"              : a 3-tuple containing the profiling, validation and attack keys in respective order.
        }
    """
    h5_file = h5py.File(file_path, "r")

    # Load the traces.
    profiling_set = np.array(h5_file['Profiling_traces/traces'], dtype=np.float64)
    attack_set = np.array(h5_file['Attack_traces/traces'], dtype=np.float64)

    # Load the plaintext.
    profiling_plaintext = h5_file['Profiling_traces/metadata']['plaintext']
    attack_plaintext = h5_file['Attack_traces/metadata']['plaintext']

    # Load the key.
    profiling_key = h5_file['Profiling_traces/metadata']['key']
    attack_key = h5_file['Attack_traces/metadata']['key']

    # Create the ciphertext.
    profiling_ciphertext = np.zeros((num_profiling_traces, len(profiling_plaintext[0])))
    attack_ciphertext = np.zeros((num_attack_traces, len(attack_plaintext[0])))

    # Overwrite the ciphertext if output/decryption.
    if "ciphertext" in h5_file['Profiling_traces/metadata'].dtype.fields:
        profiling_ciphertext = h5_file['Profiling_traces/metadata']['ciphertext']
        attack_ciphertext = h5_file['Attack_traces/metadata']['ciphertext']

    # Close h5 file.
    h5_file.close()

    # From the profiling set and the attack set, take the amount of traces and the range of features specified.
    X_profiling = profiling_set[:num_profiling_traces, first_feature:first_feature + num_features]
    X_attack = attack_set[:num_attack_traces, first_feature:first_feature + num_features]

    # Limit the plaintexts to the amount of traces specified.
    profiling_plaintext = profiling_plaintext[:num_profiling_traces]
    attack_plaintext = attack_plaintext[:num_attack_traces]

    # Limit the ciphertexts to the amount of traces specified.
    profiling_ciphertext = profiling_ciphertext[:num_profiling_traces]
    attack_ciphertext = attack_ciphertext[:num_attack_traces]

    # Limit the key to the amount of traces specified.
    profiling_key = profiling_key[:num_profiling_traces]
    attack_key = attack_key[:num_attack_traces]

    # Create validation traces.
    X_validation = X_attack[:int(num_attack_traces / 2)]

    # Remove validation traces from attack traces.
    X_attack = X_attack[int(num_attack_traces / 2):]

    # Create the validation plaintext, ciphertext and key.
    validation_plaintext = attack_plaintext[:int(num_attack_traces / 2)]
    validation_ciphertext = attack_ciphertext[:int(num_attack_traces / 2)]
    validation_key = attack_key[:int(num_attack_traces / 2)]

    # Remove validation plaintext, ciphertext and key from the attack plaintext, ciphertext and key respectively.
    attack_plaintext = attack_plaintext[int(num_attack_traces / 2):]
    attack_ciphertext = attack_ciphertext[int(num_attack_traces / 2):]
    attack_key = attack_key[int(num_attack_traces / 2):]

    read_dataset_dict = {
        "x_profiling": X_profiling,
        "x_validation": X_validation,
        "x_attack": X_attack,
        "plaintexts_profiling": profiling_plaintext,
        "plaintexts_validation": validation_plaintext,
        "plaintexts_attack": attack_plaintext,
        "ciphertexts_profiling": profiling_ciphertext,
        "ciphertexts_validation": validation_ciphertext,
        "ciphertexts_attack": attack_ciphertext,
        "keys_profiling": profiling_key,
        "keys_validation": validation_key,
        "keys_attack": attack_key,
    }

    return read_dataset_dict


def write_h5(traces_file, plaintext_file, ciphertext_file, profiling_key_file, mask_file=None, name_new_dataset="new_dataset.h5",
             profiling_attack_split=0.9, attack_key_file=None):
    """
    Write method for writing files in the h5 format as specified by the paper of the ASCAD dataset from .txt files as
    used by the aes_hd_mm dataset: https://chest.coe.neu.edu/?current_page=POWER_TRACE_LINK&software=ptmasked.

    :param traces_file            : path to the traces file.
    :param plaintext_file         : path to the plaintext file.
    :param ciphertext_file        : path to the ciphertext file.
    :param profiling_key_file     : path to the profiling key and attack key if no attack key specified.
    :param mask_file              : path to the mask file, if applicable.
    :param name_new_dataset       : name, and path, of the new dataset created, if specified.
    :param profiling_attack_split : percentage of the traces that will be used for profiling and attacking, 0.9 by default.
    :param attack_key_file        : path to the attack key file, if applicable.

    :return:
        No object or data, but creates and writes the file with the specified 'name_new_dataset' name.
    """

    # Load traces, plaintext and ciphertext from file.
    traces = np.array(np.loadtxt(traces_file))
    plaintext = np.array(np.loadtxt(plaintext_file, dtype=np.uint8))
    ciphertext = np.array(np.loadtxt(ciphertext_file, dtype=np.uint8))

    # Depending on how the profiling key is supplied, configure the profiling key.
    profiling_key_raw = np.array(np.loadtxt(profiling_key_file, dtype=np.uint8, converters={_: lambda s: int(s, 16) for _ in range(16)}),
                                 dtype=np.uint8)
    profiling_key = np.zeros((len(traces), len(profiling_key_raw)), dtype=np.uint8)
    if profiling_key_raw.ndim == 1:
        for i in range(len(profiling_key)):
            profiling_key[i] = profiling_key_raw
    else:
        profiling_key = profiling_key_raw

    # Define attack_key, approach it the same way as the profiling key if it is supplied, otherwise copy profiling key.
    attack_key = None
    if attack_key_file is None:
        attack_key = profiling_key
    else:
        attack_key_raw = np.loadtxt(profiling_key_file, dtype=np.uint8, converters={_: lambda s: int(s, 16) for _ in range(16)})
        attack_key = np.zeros((len(traces), len(profiling_key_raw)), dtype=np.uint8)
        if attack_key_raw.ndim == 1:
            for i in range(len(attack_key)):
                attack_key[i] = attack_key_raw
        else:
            attack_key = attack_key_raw

    # Define and split mask if applicable.
    profiling_mask = None
    attack_mask = None
    if mask_file is not None:
        mask = np.loadtxt(mask_file)

        profiling_mask = mask[:int(len(mask) * profiling_attack_split)]
        attack_mask = mask[int(len(mask) * profiling_attack_split):]

    # Split traces according to profiling_attack_split.
    profiling_traces = traces[:int(len(traces) * profiling_attack_split)]
    attack_traces = traces[int(len(traces) * profiling_attack_split):]

    # Split plaintext according to profiling_attack_split.
    profiling_plaintext = plaintext[:int(len(plaintext) * profiling_attack_split)]
    attack_plaintext = plaintext[int(len(plaintext) * profiling_attack_split):]

    # Split ciphertext according to profiling_attack_split.
    profiling_ciphertext = ciphertext[:int(len(ciphertext) * profiling_attack_split)]
    attack_ciphertext = ciphertext[int(len(ciphertext) * profiling_attack_split):]

    write_h5_direct_data_(name_new_dataset, profiling_traces, attack_traces, profiling_plaintext, attack_plaintext,
                          profiling_ciphertext, attack_ciphertext, profiling_mask, attack_mask,
                          profiling_key, attack_key)


def write_h5_direct_data_(path_to_h5, profiling_traces, attack_traces, profiling_plaintext, attack_plaintext,
                          profiling_ciphertext, attack_ciphertext, profiling_mask, attack_mask,
                          profiling_key, attack_key):
    """
        Helper method for writing data in the .h5 format, this method takes direct data and therefore can be used in
        multiple different methods.

        :param path_to_h5           : the path to the h5 to be written.
        :param profiling_traces     : the profiling traces to be used.
        :param attack_traces        : the attack traces to be used.
        :param profiling_plaintext  : the profiling plaintext to be used.
        :param attack_plaintext     : the attack plaintext to be used.
        :param profiling_ciphertext : the profiling ciphertext to be used.
        :param attack_ciphertext    : the attack ciphertext to be used.
        :param profiling_mask       : the profiling mask to be used.
        :param attack_mask          : the attack mask to be used.
        :param profiling_key        : the profiling key to be used.
        :param attack_key           : the attack key to be used.

        :return:
            No object or data, but creates and writes the file with the specified 'path_to_h5' name.
        """
    # Define outfile.
    out_file = h5py.File(path_to_h5, 'w')

    # Define the profiling and attack group.
    profiling_traces_group = out_file.create_group("Profiling_traces")
    attack_traces_group = out_file.create_group("Attack_traces")

    # Create the profiling and attack traces datasets in their respective groups.
    profiling_traces_group.create_dataset(name="traces", data=profiling_traces, dtype=profiling_traces.dtype)
    attack_traces_group.create_dataset(name="traces", data=attack_traces, dtype=attack_traces.dtype)

    # Create the metadata types for the metadata.
    metadata_type_profiling = np.dtype([("plaintext", profiling_plaintext.dtype, (len(profiling_plaintext[0]),)),
                                        ("ciphertext", profiling_ciphertext.dtype, (len(profiling_ciphertext[0]),)),
                                        ("key", profiling_key.dtype, (len(profiling_key[0]),)),
                                        ("mask", profiling_mask.dtype, (len(profiling_mask[0]),))])
    metadata_type_attack = np.dtype([("plaintext", attack_plaintext.dtype, (len(attack_plaintext[0]),)),
                                     ("ciphertext", attack_ciphertext.dtype, (len(attack_ciphertext[0]),)),
                                     ("key", attack_key.dtype, (len(attack_key[0]),)),
                                     ("mask", attack_mask.dtype, (len(attack_mask[0]),))])

    # TODO: Add profiling/attack indexing if desired.
    # Create profiling metadata dataset.
    profiling_metadata = np.array(
        [(profiling_plaintext[n], profiling_ciphertext[n], profiling_key[n], profiling_mask[n]) for n, k in
         zip(range(len(profiling_traces)), range(len(profiling_traces)))], dtype=metadata_type_profiling)
    profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

    # Create attack metadata dataset.
    attack_metadata = np.array([(attack_plaintext[n], attack_ciphertext[n], attack_key[n], attack_mask[n]) for n, k in
                                zip(range(len(attack_traces)), range(len(attack_traces)))], dtype=metadata_type_attack)
    attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

    # Flush and close the output file.
    out_file.flush()
    out_file.close()


def update_traces_h5(path_to_h5, traces_file=None, plaintext_file=None, ciphertext_file=None, mask_file=None,
                     profiling_key_file=None, attack_key_file=None, name_updated_dataset=None, profiling_attack_split=0.9):
    """
    Write method for writing files in the h5 format as specified by the paper of the ASCAD dataset from .txt files as
    used by the aes_hd_mm dataset: https://chest.coe.neu.edu/?current_page=POWER_TRACE_LINK&software=ptmasked.

    :param path_to_h5             : path to the .h5 file to be updated.
    :param traces_file            : path to the traces file.
    :param plaintext_file         : path to the plaintext file.
    :param ciphertext_file        : path to the ciphertext file.
    :param mask_file              : path to the mask file, if applicable.
    :param profiling_key_file     : path to the profiling key.
    :param attack_key_file        : path to the attack key.
    :param name_updated_dataset   : if specified the original .h5 won't be overwritten but a new one with the name 'name_updated_dataset' will be created.
    :param profiling_attack_split : percentage of the traces that will be used for profiling and attacking, 0.9 by default.

    :return:
        No object or data, but creates and writes the file with the specified 'name_new_dataset' name.
    """

    in_file = h5py.File(path_to_h5, "r")

    if traces_file is None:
        profiling_traces = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
        attack_traces = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
    else:
        # Load traces.
        traces = np.array(np.loadtxt(traces_file))

        # Split traces according to profiling_attack_split.
        profiling_traces = traces[:int(len(traces) * profiling_attack_split)]
        attack_traces = traces[int(len(traces) * profiling_attack_split):]

    if plaintext_file is None:
        profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
        attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
    else:
        # Load plaintext.
        plaintext = np.array(np.loadtxt(plaintext_file, dtype=np.uint8))

        # Split plaintext according to profiling_attack_split.
        profiling_plaintext = plaintext[:int(len(plaintext) * profiling_attack_split)]
        attack_plaintext = plaintext[int(len(plaintext) * profiling_attack_split):]

    if ciphertext_file is None:
        profiling_ciphertext = in_file['Profiling_traces/metadata']['ciphertext']
        attack_ciphertext = in_file['Attack_traces/metadata']['ciphertext']
    else:
        # Load ciphertext.
        ciphertext = np.array(np.loadtxt(ciphertext_file, dtype=np.uint8))

        # Split ciphertext according to profiling_attack_split.
        profiling_ciphertext = ciphertext[:int(len(ciphertext) * profiling_attack_split)]
        attack_ciphertext = ciphertext[int(len(ciphertext) * profiling_attack_split):]

    if mask_file is None:
        profiling_mask = in_file['Profiling_traces/metadata']['masks']
        attack_mask = in_file['Attack_traces/metadata']['masks']
    else:
        # Define and split mask if applicable.
        mask = np.loadtxt(mask_file)

        profiling_mask = mask[:int(len(mask) * profiling_attack_split)]
        attack_mask = mask[int(len(mask) * profiling_attack_split):]

    if profiling_key_file is None:
        profiling_key = in_file['Profiling_traces/metadata']['key']
    else:
        # Depending on how the profiling key is supplied, configure the profiling key.
        profiling_key_raw = np.array(
            np.loadtxt(profiling_key_file, dtype=np.uint8, converters={_: lambda s: int(s, 16) for _ in range(16)}),
            dtype=np.uint8)
        profiling_key = np.zeros((len(traces), len(profiling_key_raw)), dtype=np.uint8)
        if profiling_key_raw.ndim == 1:
            for i in range(len(profiling_key)):
                profiling_key[i] = profiling_key_raw
        else:
            profiling_key = profiling_key_raw

    if attack_key_file is None:
        attack_key = in_file['Attack_traces/metadata']['key']
    else:
        # Define attack_key, approach it the same way as the profiling key.
        attack_key_raw = np.loadtxt(profiling_key_file, dtype=np.uint8,
                                    converters={_: lambda s: int(s, 16) for _ in range(16)})
        attack_key = np.zeros((len(traces), len(profiling_key_raw)), dtype=np.uint8)
        if attack_key_raw.ndim == 1:
            for i in range(len(attack_key)):
                attack_key[i] = attack_key_raw
        else:
            attack_key = attack_key_raw

    write_h5_direct_data_(name_updated_dataset, profiling_traces, attack_traces, profiling_plaintext, attack_plaintext,
                          profiling_ciphertext, attack_ciphertext, profiling_mask, attack_mask,
                          profiling_key, attack_key)


def merge_group_h5(f1_gp, f2_gp, fout_gp):
    """
    This function merges two h5 groups f1_gp and f2_gp in a single group fout_gp. It assumes that the groups are
    formed of arrays of the same name. For each entry name k in f1_gp and f2_gp, an array that results in the
    concatenation of the arrays f1_gp[k] and f2_gp[k] is created in fout_gp. The arrays are concatenated along their
    first axis.

    :param f1_gp   : the first group to be merged.
    :param f2_gp   : the second group to be merged.
    :param fout_gp : the output group.

    :return:
        Nothing, modifies the 'fout_gp'.
    """
    keys = f1_gp.keys()
    dtypes = {}
    shapes = {}
    for k in keys:
        dtypes[k] = f1_gp[k].dtype
        shapes[k] = (f1_gp[k].shape[0] + f2_gp[k].shape[0],) + f1_gp[k].shape[1:]
    for k in keys:
        fout_gp.create_dataset(k, shapes[k], dtype=dtypes[k])
        for i in range(shapes[k][0]):
            if i < f1_gp[k].shape[0]:
                fout_gp[k][i] = f1_gp[k][i]
            else:
                fout_gp[k][i] = f2_gp[k][i - f1_gp[k].shape[0]]


def merge_files_h5(file1, file2, out_file_name="fileout"):
    """
    This function merges the arrays contained in the .h5 files file1 and file2, resulting in a new .h5 file named
    "fileout" by default. The arrays are concatenated along their first axis.

    :param file1         : the first file to be merged.
    :param file2         : the second file to be merged.
    :param out_file_name : the name of the output file.

    :return:
        No object or data, but creates and writes the file with the specified 'out_file_name' name.
    """
    f1 = h5py.File(file1, "r")
    f2 = h5py.File(file2, "r")
    fout = h5py.File(out_file_name, "w")

    print("Concatenation Profiling_traces")
    fout_profile_gp = fout.create_group("Profiling_traces")
    merge_group_h5(f1["Profiling_traces"], f2["Profiling_traces"], fout_profile_gp)

    print("Concatenation Attack_traces")
    fout_attack_gp = fout.create_group("Attack_traces")
    merge_group_h5(f1["Attack_traces"], f2["Attack_traces"], fout_attack_gp)

    fout.flush()
    fout.close()
    f1.close()
    f2.close()
