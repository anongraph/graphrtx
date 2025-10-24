#!/usr/bin/env python3

import itertools
import subprocess
import sys

from subprocess import DEVNULL


# refer to default_test_configuration.h for documentation
default_config = {
    "PERPENDICULAR_RAYS": 1,
    "COMPACTION": 1,
    "FORCE_SINGLE_ANYHIT": 1,
    "USE_CLOSESTHIT_INSTEAD_OF_ANYHIT": 0,
    "START_RAY_AT_ZERO": 0,
    "LARGE_KEYS": 0,
    "LEAVE_GAPS_FOR_MISSES": 0,
    "MULTIPLE_HITS_PER_RAY": 0,
    "SKIP_PROBING": 0,
    "FORCE_UNIFORM_KEYS": 0,

    "UPDATE_TYPE": 0,

    "PRIMITIVE": 0,

    "INSERT_SORTED": 0,
    "PROBE_SORTED": 0,

    "EXPONENT_BIAS": 0,

    "NUM_BUILD_KEYS_LOG": 26,
    "NUM_PROBE_KEYS_LOG": 27,
    "NUM_RAYS_PER_THREAD_LOG": 0,
    "KEY_REPLICATION_LOG": 0,
    "RANGE_QUERY_HIT_COUNT_LOG": 0,
    "NUM_UPDATES_LOG": 0,
    "LOCAL_UPDATE_CHUNK_SIZE_LOG": 1,
    "MISS_PERCENTAGE": 0,
    "OUT_OF_RANGE_PERCENTAGE": 0,
    "KEY_STRIDE_LOG": 0,

    "INT_TO_FLOAT_CONVERSION_MODE": 3,

    "LARGE_KEY_DECOMPOSITION": 23_23_18,
    "SMALL_KEY_DECOMPOSITION": 23_09_00,
}

experiments = {
    "baseline": {
        "INSERT_SORTED": [0, 1],
        "PROBE_SORTED": [0, 1],
    },
    "update-then-probe": {
        "UPDATE_TYPE": [0, 1, 2],
        "NUM_UPDATES_LOG": [0, 2, 4, 6, 8, 10, 12],
    },
    "update-locality": {
        "UPDATE_TYPE": [2],
        "NUM_UPDATES_LOG": [24],
        "LOCAL_UPDATE_CHUNK_SIZE_LOG": [1, 2, 4, 6, 8, 10, 12],
    },
    "update-only": {
        "UPDATE_TYPE": [0, 1, 2],
        "INSERT_SORTED": [0, 1],
        "NUM_UPDATES_LOG": [0, 2, 4, 6, 8, 10, 12, 16, 20, 24],
        "SKIP_PROBING": [1],
    },
    "using-closest-hit": {
        "PROBE_SORTED": [0, 1],
        "USE_CLOSESTHIT_INSTEAD_OF_ANYHIT": [0, 1],
    },
    "int-to-ray": {
        "NUM_BUILD_KEYS_LOG": [21, 22, 23, 24, 25, 26],
        "PERPENDICULAR_RAYS": [0, 1],
        "START_RAY_AT_ZERO": [1],
        "INT_TO_FLOAT_CONVERSION_MODE": [0, 1, 2, 3],
    },
    "stride": {
        "NUM_BUILD_KEYS_LOG": [21, 22, 23, 24, 25, 26],
        "START_RAY_AT_ZERO": [1],
        "INT_TO_FLOAT_CONVERSION_MODE": [0, 1, 2, 3],
        "KEY_STRIDE_LOG": [0, 1, 2, 3],
    },
    "key-size": {
        "LARGE_KEYS": [0, 1],
    },
    "exponent-shift": {
        "NUM_BUILD_KEYS_LOG": [21, 22, 23, 24, 25, 26],
        "START_RAY_AT_ZERO": [1],
        "INT_TO_FLOAT_CONVERSION_MODE": [0, 1, 2, 3],
        "EXPONENT_BIAS": [-20, -10, 0, 10, 20],
    },
    "primitive": {
        "NUM_BUILD_KEYS_LOG": [21, 22, 23, 24, 25, 26],
        "PROBE_SORTED": [0, 1],
        "COMPACTION": [0, 1],
        "PRIMITIVE": [0, 1, 2],
        "SMALL_KEY_DECOMPOSITION": [22_10_00],
    },
    "ray-properties": {
        "PERPENDICULAR_RAYS": [0, 1],
        "START_RAY_AT_ZERO": [0, 1],
        "PROBE_SORTED": [0, 1],
    },
    "ordering": {
        "INSERT_SORTED": [-1, 0, 1],
        "PROBE_SORTED": [-1, 0, 1, 2],
    },
    "key-decomposition-xy": {
        "NUM_BUILD_KEYS_LOG": [20, 26],
        "FORCE_UNIFORM_KEYS": [0, 1],
        "SMALL_KEY_DECOMPOSITION": [23_09_00, 22_10_00, 21_11_00, 20_12_00, 19_13_00, 18_14_00, 17_15_00, 16_16_00],
    },
    "key-decomposition-xz": {
        "NUM_BUILD_KEYS_LOG": [20, 26],
        "FORCE_UNIFORM_KEYS": [0, 1],
        "SMALL_KEY_DECOMPOSITION": [23_00_09, 22_00_10, 21_00_11, 20_00_12, 19_00_13, 18_00_14, 17_00_15, 16_00_16],
    },
    "key-decomposition-uniform": {
        "LARGE_KEYS": [1],
        "FORCE_UNIFORM_KEYS": [1],
        "LARGE_KEY_DECOMPOSITION": [23_23_18, 23_18_23, 18_23_23, 22_21_21, 21_22_21, 21_21_22],
    },
    "multiple-rays-per-thread": {
        "PROBE_SORTED": [0, 1],
        "NUM_RAYS_PER_THREAD_LOG": [0, 1, 2, 3, 6, 9, 12, 15, 18],
    },
    "range-query-start-ray-at-zero": {
        "RANGE_QUERY_HIT_COUNT_LOG": [0, 2, 4, 6, 8, 10, 12],
        "START_RAY_AT_ZERO": [0, 1],
        "MULTIPLE_HITS_PER_RAY": [1],
    },
    "range-query-ordering": {
        "RANGE_QUERY_HIT_COUNT_LOG": [0, 2, 4, 6, 8, 10, 12],
        "PROBE_SORTED": [0, 2],
        "MULTIPLE_HITS_PER_RAY": [1],
    },
    "range-query-conversion-mode": {
        "RANGE_QUERY_HIT_COUNT_LOG": [0, 2, 4, 6, 8],
        "START_RAY_AT_ZERO": [1],
        "INT_TO_FLOAT_CONVERSION_MODE": [0, 1, 2, 3],
        "MULTIPLE_HITS_PER_RAY": [1],
    },
    "range-query-key-decomposition": {
        "NUM_PROBE_KEYS_LOG": [21],
        "RANGE_QUERY_HIT_COUNT_LOG": [8, 10],
        "MULTIPLE_HITS_PER_RAY": [1],
        "SMALL_KEY_DECOMPOSITION": [23_09_00, 22_10_00, 21_11_00, 20_12_00, 19_13_00, 18_14_00, 17_15_00, 16_16_00],
    },
    "miss": {
        "PROBE_SORTED": [0, 2],
        "LEAVE_GAPS_FOR_MISSES": [1],
        "MISS_PERCENTAGE": [0, 50, 90, 99, 100],
    },
    "miss-out-of-range": {
        "PROBE_SORTED": [0, 2],
        "LEAVE_GAPS_FOR_MISSES": [1],
        "OUT_OF_RANGE_PERCENTAGE": [0, 50, 90, 99, 100],
    },
}

if len(sys.argv) > 1:
    experiments = {name: experiments[name] for name in sys.argv[1:]}

print("MAKE SURE TO RUN THIS SCRIPT FROM THE BUILD DIRECTORY, I.E., THE ONE WITH THE MAKEFILE!\n")

for description, configurations in experiments.items():

    # delete result file
    output_file = f"{description}.csv"
    subprocess.run(["rm", output_file])

    with open(output_file, "w+") as output_file:
        replicate_key = [[(key, value) for value in values] for key, values in configurations.items()]
        all_combinations = [dict(test_config) for test_config in itertools.product(*replicate_key)]

        for it, test_config in enumerate(all_combinations):
            assert default_config.keys() >= test_config.keys()
            modified_config = {**default_config, **test_config}

            print(f"EXPERIMENT '{description}' ({it * 100 // len(all_combinations)}%)\n")
            generated_header = [f"#define {key} {value}\n" for key, value in modified_config.items()]
            print("".join(generated_header), file=sys.stderr, flush=True)

            with open(f"../src/test_configuration_override.h", "w") as header_file:
                header_file.writelines(generated_header)
            # build and run
            subprocess.run(["make", "-B", "-j16"], stdout=DEVNULL, stderr=DEVNULL).check_returncode()
            try:
                subprocess.run(["./run_experiment"], stdout=output_file, timeout=300).check_returncode()
            except subprocess.TimeoutExpired:
                pass

            print("", file=sys.stderr, flush=True)
