# %%
import base64
import gzip
import itertools
import json
import re

from greenery import Charclass
from greenery.fsm import Fsm
from greenery.rxelems import from_fsm
from tqdm import tqdm

CHARCLASSES = [
    Charclass("a"),  # a
    ~Charclass("a"),  # b
]
TEST_INPUTS = ["".join(inp) for inp_len in range(20) for inp in itertools.product("ab", repeat=inp_len)]


def generate_regexes(n_states: int) -> set[str]:
    initial_state = 0

    state_transition_maps: list[dict[int, dict[Charclass, int]]] = []
    # Generate all possible state transition combinations
    for transition_values in itertools.product(range(n_states), repeat=n_states * len(CHARCLASSES)):
        transition_map = {}
        for state in range(n_states):
            transition_map[state] = {}
            for i, charclass in enumerate(CHARCLASSES):
                next_state = transition_values[state * len(CHARCLASSES) + i]
                transition_map[state][charclass] = next_state
        state_transition_maps.append(transition_map)

    print(f"Generated {len(state_transition_maps)} {n_states}-state transition maps")

    all_final_states = [
        final_states for k in range(1, n_states + 1) for final_states in itertools.combinations(range(n_states), k)
    ] + [tuple()]
    fsms = []
    for final_states in all_final_states:
        for state_transition_map in state_transition_maps:
            fsms.append(
                Fsm(
                    alphabet=CHARCLASSES,
                    states=list(range(n_states)),
                    initial=initial_state,
                    finals=final_states,
                    map=state_transition_map,
                )
            )
    print(f"Generated {len(fsms)} {n_states}-state FSMs. Reducing and converting to regexes...")

    regexes = set(str(from_fsm(fsm.reduce())) for fsm in tqdm(fsms))
    print(f"Left with {len(regexes)} {n_states}-state unique regexes")

    return regexes


def calculate_behavior(regexes: set[str]) -> dict[str, list[bool]]:
    print(f"Getting behavior for {len(regexes)} regexes")
    compiled_regexes = [re.compile(r) for r in regexes]
    return {rs: [bool(rc.fullmatch(s)) for s in TEST_INPUTS] for rs, rc in tqdm(list(zip(regexes, compiled_regexes)))}


def fix_formatting(regexes: set[str]) -> set[str]:
    # Greenery uses [] for match-nothing, but that only seems to be supported in JS
    return set(r.replace("[^a]", "b").replace("[]", r"(?!)") for r in regexes)


def deduplicate_regexes(regex_behavior: dict[str, list[bool]]) -> set[str]:
    print(f"Deduplicating {len(regex_behavior)} regexes")

    duplicates = set()
    for re1, re2 in itertools.product(regex_behavior.keys(), repeat=2):
        if re1 != re2 and regex_behavior[re1] == regex_behavior[re2]:
            duplicates.add(re1 if len(re1) > len(re2) else re2)
    print(f"Found {len(duplicates)} duplicates")

    return set(regex_behavior.keys()) - duplicates


def store_regexes(regexes: set[str], regex_behavior: dict[str, list[bool]], two_state_regexes: set[str]):
    regex_info = {}

    print("Storing regexes")
    for r in tqdm(regexes):
        regex_info[r] = {
            "states": 2 if r in two_state_regexes else 3,
            "match_count": regex_behavior[r].count(True),
            "match_bitmap": compress_encode(regex_behavior[r]),
        }

    with open("regex_info.json", "w") as f:
        json.dump(regex_info, f)


def compress_encode(bool_array):
    # Convert to bitmap
    bitmap = "".join("1" if b else "0" for b in bool_array)
    # Compress
    compressed = gzip.compress(bitmap.encode("utf-8"))
    # Encode as base64
    return base64.b64encode(compressed).decode("ascii")


def compress_decode(encoded_str):
    # Decode base64
    compressed = base64.b64decode(encoded_str)
    # Decompress
    bitmap = gzip.decompress(compressed).decode("utf-8")
    # Convert back to boolean array
    return [b == "1" for b in bitmap]


if __name__ == "__main__":
    two_state_regexes = fix_formatting(generate_regexes(n_states=2))
    three_state_regexes = fix_formatting(generate_regexes(n_states=3))

    print(
        f"Generated {len(two_state_regexes)} two-state regexes and {len(three_state_regexes)} three-state regexes,"
        f" with {len(two_state_regexes & three_state_regexes)} overlaps"
    )

    regex_behavior = calculate_behavior(two_state_regexes | three_state_regexes)
    deduplicated_regexes = deduplicate_regexes(regex_behavior)
    store_regexes(deduplicated_regexes, regex_behavior, two_state_regexes)

    print(f"Stored {len(deduplicated_regexes)} regexes")
