import os
import json
import time
from typing import List, Dict, Any

import re

class PopulationDatabase:

    def __init__(self, root_folder: str, keep_best_k: int = 10):
        self.root_folder = root_folder
        self.keep_best_k = keep_best_k

        self.pop_folder = os.path.join(root_folder, "pops")
        self.pool_pop_folder = os.path.join(root_folder, "pool_pops")
        self.best_folder = os.path.join(root_folder, "pops_best")
        self.pool_best_folder = os.path.join(root_folder, "pool_pops_best")
        self.programs_folder = os.path.join(root_folder, "all_programs")
        self.pools_folder = os.path.join(root_folder, "all_pools")

        os.makedirs(self.pop_folder, exist_ok=True)
        os.makedirs(self.pool_pop_folder, exist_ok=True)
        os.makedirs(self.best_folder, exist_ok=True)
        os.makedirs(self.pool_best_folder, exist_ok=True)
        os.makedirs(self.programs_folder, exist_ok=True)
        os.makedirs(self.pools_folder, exist_ok=True)

    # -----------------------
    # POPULATION MANAGEMENT
    # -----------------------
    def save_population(
            self,
            programs: List[Dict[str, Any]],
            generation: int,
            is_best: bool = False,
            only_best: bool = False,
    ):

        programs_sorted = sorted(
            programs,
            key=lambda p: (
                p.get("offspring", {}).get("objective") is None,
                p.get("offspring", {}).get("objective")
            ),
            reverse=True  # set to False if lower objective is better
        )

        # Keep only best program if requested
        if only_best and programs_sorted:
            programs_to_save = [programs_sorted[0]]
        else:
            programs_to_save = programs_sorted

        folder = self.best_folder if is_best else self.pop_folder
        file_path = os.path.join(folder, f"population_gen_{generation}.json")
        with open(file_path, "w+") as f:
            json.dump(programs_to_save, f, indent=4)

    def load_population(self, generation: int, best: bool = False) -> List[Dict[str, Any]]:
        folder = self.best_folder if best else self.pop_folder
        file_path = os.path.join(folder, f"population_gen_{generation}.json")

        if not os.path.exists(file_path):
            return []

        with open(file_path, "r") as f:
            return json.load(f)

    def save_pools(self, pools: List[Dict[str, Any]], generation: int, only_best: bool = False):
        """
        Save pools to file. Can either save only the best pool or all pools
        ordered from best to worst.

        - When saving all pools to one file, the key 'evaluation_results' is excluded.
        - Existing individual pool files are skipped.
        - Unevaluated pools (without pool_score) are also saved but listed last.
        """
        if not pools:
            print("[PopulationDatabase] No pools to save.")
            return

        # Separate evaluated and unevaluated pools
        evaluated = [p for p in pools if p.get("pool_score") is not None]
        unevaluated = [p for p in pools if p.get("pool_score") is None]

        # Sort evaluated ones (best → worst)
        evaluated_sorted = sorted(evaluated, key=lambda x: x["pool_score"])

        # Combine (evaluated first, unevaluated last)
        combined_pools = evaluated_sorted + unevaluated

        if only_best:
            if not evaluated_sorted:
                print("[PopulationDatabase] No evaluated pools found — cannot save best pool.")
                return

            best_pool = evaluated_sorted[0]
            file_path = os.path.join(self.pool_best_folder, f"best_pool_gen_{generation}.json")
            with open(file_path, "w+") as f:
                json.dump(best_pool, f, indent=4)
            print(f"[PopulationDatabase] Saved best pool → {file_path}")
            return

        # Clean up and save all pools (excluding heavy fields)
        cleaned_pools = [
            {k: v for k, v in pool.items() if k != "evaluation_results"} for pool in combined_pools
        ]

        file_path = os.path.join(self.pool_pop_folder, f"pools_gen_{generation}.json")
        with open(file_path, "w+") as f:
            json.dump(cleaned_pools, f, indent=4)

        print(
            f"[PopulationDatabase] Saved {len(combined_pools)} pools (evaluated first, unevaluated last) → {file_path}")

        # ---- Save each pool individually ----
        for pool in combined_pools:
            pool_id = pool.get("pool_id", f"unknown_{time.time()}")
            pool_path = os.path.join(self.pools_folder, f"{pool_id}.json")

            # Skip if already exists
            if os.path.exists(pool_path):
                continue

            pool_data = pool if isinstance(pool, dict) else pool.__dict__
            with open(pool_path, "w+") as f:
                json.dump(pool_data, f, indent=4)

        print(f"[PopulationDatabase] Saved individual pool files to {self.pools_folder}")


    # -----------------------
    # BEST POOL HANDLING
    # -----------------------
    def update_best_pools(self, pools: List[Dict[str, Any]], generation: int):
        # Load existing best archive
        best_pools = []
        for gen_file in os.listdir(self.best_folder):
            if gen_file.endswith(".json"):
                with open(os.path.join(self.best_folder, gen_file), "r") as f:
                    best_pools.extend(json.load(f))

        # Merge and select top-k by pool_score
        combined = best_pools + pools
        combined = [p for p in combined if p.get("pool_score") is not None]

        sorted_pools = sorted(combined, key=lambda x: x["pool_score"], reverse=True)
        top_pools = sorted_pools[: self.keep_best_k]

        # Save as "population_gen_<generation>.json"
        self.save_population(top_pools, generation, is_best=True)

    # -----------------------
    # HEURISTIC ARCHIVE
    # -----------------------
    def load_programs_until(self, pop_x: int) -> List[Dict[str, Any]]:
        """
        Load all program files up to population 'pop_x' (inclusive).
        """
        all_programs = []
        #pattern = re.compile(r"program_pop_(\d+)_op_.*\.json")
        pattern = re.compile(r"program_pool_op_.*\._pop_(\d+)_.*\..json")

        for file_name in os.listdir(self.programs_folder):
            if "Exception" in file_name:
                continue
            match = pattern.match(file_name)
            if match:
                pop_num = int(match.group(1))
                if pop_num <= pop_x:
                    file_path = os.path.join(self.programs_folder, file_name)
                    with open(file_path, "r") as f:
                        all_programs.extend(json.load(f))
        return all_programs

    def load_programs_only(self, pop_x: int) -> List[Dict[str, Any]]:
        """
        Load only program files for population 'pop_x'.
        """
        all_programs = []
        pattern = re.compile(r"_pop_(\d+)_")

        for file_name in os.listdir(self.programs_folder):
            if "Exception" in file_name:
                print("Skip Exception file ", file_name)
                continue
            match = pattern.search(file_name)
            if match:
                pop_num = int(match.group(1))
                if pop_num == pop_x:
                    file_path = os.path.join(self.programs_folder, file_name)
                    with open(file_path, "r") as f:
                        all_programs.append(json.load(f))
        return all_programs

    def save_program(
        self,
        current_file,
        file_name: str,
        possible_exception: str = ""):

        add_exception_string = "" if possible_exception == "" else "_Exception"
        with open(os.path.join(self.programs_folder, f'program_{file_name}{add_exception_string}.json'), 'w') as file:
            json.dump(current_file, file, indent=4)

