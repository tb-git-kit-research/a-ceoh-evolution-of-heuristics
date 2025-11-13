import datetime
import random
from collections import defaultdict
from typing import List

from solver.ceoh.methods.eoh_pooling.database import PopulationDatabase
from solver.ceoh.methods.eoh_pooling.util import get_agv_weight, get_destroy_scores

import trace

#########################################################
# Pool Settings
#########################################################

# TODO: Move to paras
POOL_SIZE = 4
NR_OF_POOLS = 20
NR_OF_POOL_MUTATIONS = int(NR_OF_POOLS / 2)
NR_OF_MUTATIONS = int(POOL_SIZE / 2)

class Pool:
    def __init__(self, programs):

        if programs is None or len(programs) == 0:
            raise ValueError("CEOH_POOLING: Failed to initalize pool!")

        self._pool_id = None

        self._programs = programs
        self._ids = self.get_ids()

        self._pool_score = None
        self._pool_reference = None
        self._pool_gap = None
        self._individual_score = [None] * len(programs)

        self._evaluation_results = None

        self._keep = False

    #########################################################
    # Getter and Setter
    #########################################################

    def set_evaluation_results(self, results):
        self._evaluation_results = results

    def get_evaluation_results(self):
        return self._evaluation_results

    def set_pool_id(self, pool_id):
        self._pool_id = pool_id

    def get_pool_id(self):
        return self._pool_id

    def get_score(self):
        return self._pool_score

    def set_pool_score(self, score):
        self._pool_score = score

    def set_pool_reference(self, score):
        self._pool_reference = score

    def set_pool_gap(self, score):
        self._pool_gap = score

    def set_program_score(self, index, score):
        self._individual_score[index] = score

    def get_program_scores(self):
        return self._individual_score

    def get_codes(self):
        return [p["offspring"]["code"] for p in self._programs]

    def get_programs(self):
        return self._programs

    def get_offsprings(self):
        return [p["offspring"] for p in self._programs]

    def get_lowest_score_program_index(self):

        return self._individual_score.index(min(x for x in self._individual_score if x is not None))

    def set_keep(self, val: bool):
        self._keep = val

    def get_ids(self):
        return [p["offspring"]["offspring_id"] for p in self._programs]

    #########################################################
    # Utils
    #########################################################

    def keep_pool(self):
        return self._keep

    #########################################################
    # Core Functions
    #########################################################

    def update(self, program, random=False):
        if random:
            index = random.randint(0, len(self._programs) - 1)
        else:
            index = self.get_lowest_score_program_index()

        self._programs[index] = program
        self._individual_score[index] = None
        self._ids = self.get_ids()

    def create_pool_id(self, pop,nr, op):
        now = datetime.datetime.now()
        self.set_pool_id(f"pool_pop_{pop}_n{nr}_{op}_{now.strftime('%y%m%d_%H%M%S')}{now.strftime('%f')[:3]}")



# Limit number of same programs in different pools
class CeohPooling:

    def __init__(self, db: PopulationDatabase):
        self.db: PopulationDatabase = db
        self.pools: List[Pool] = []

    #########################################################
    # Getter and Setter
    #########################################################

    def get_programs_from_best_pools(self, top_x: int = NR_OF_POOLS):
        """
        Return full program dicts (not just offspring) from the top-X best pools.

        Args:
            top_x (int): Number of best pools to select (default = all pools).

        Returns:
            List of unique programs (dicts with offspring and metadata).
        """
        best_pool_indices = self.get_pool_best_pools(top_x)

        collected = []
        seen_ids = set()

        for idx in best_pool_indices:
            for program in self.pools[idx].get_programs():
                oid = program["offspring"]["offspring_id"]
                if oid not in seen_ids:
                    seen_ids.add(oid)
                    collected.append(program)

        return collected

    def get_pool_worst_pools(self, nr_of_pools):
        return self.get_sorted_pools(False)[:nr_of_pools]

    def get_pool_best_pools(self, nr_of_pools):
        return self.get_sorted_pools(True)[:nr_of_pools]

    def get_sorted_pools(self, ascending=True):
        scored_pools = [(i, p.get_score()) for i, p in enumerate(self.pools) if p.get_score() is not None]
        scored_pools.sort(key=lambda x: x[1], reverse=not ascending)
        return [i for i, _ in scored_pools]

    def set_individual_scores(self, pool_index, individual_scores: List[float]):
        for i, c in enumerate(individual_scores):
            self.pools[pool_index].set_program_score(i, c)

        #self.pools[pool_index].set_pool_score( sum(individual_scores) / len(individual_scores) )

    def get_programs_in_pools(self):
        return [pool.get_programs() for pool in self.pools]

    def get_offsprings_in_pools(self):
        return [pool.get_offsprings() for pool in self.pools]

    #########################################################
    # Initialization Pools
    #########################################################

    def add_new_pools(self, pop):
        """
        Add new pools to the existing set of pools.

        Args:
            pop: The population reference to load programs from.
        """
        programs = self.db.load_programs_only(pop)

        if len(programs) < POOL_SIZE:
            raise ValueError(f"Not enough programs ({len(programs)}) to create pools of size {POOL_SIZE}")

        if pop == 0:
            pool_number = NR_OF_POOLS
        else:
            pool_number = NR_OF_POOL_MUTATIONS

        new_pools = []
        for i in range(pool_number):
            pool_programs = random.sample(programs, POOL_SIZE)
            new_pool = Pool(pool_programs)
            new_pool.create_pool_id(pop, i, "new")
            new_pools.append(new_pool)


        # Extend existing pools
        self.pools.extend(new_pools)

    #########################################################
    # Utils
    #########################################################

    def convert_to_dict(self):
        pool_dicts = []
        for i, pool in enumerate(self.pools):
            pool_dict = {
                "pool_id": pool._pool_id,
                "programs": pool._programs,
                "pool_score": pool._pool_score,
                "pool_reference": pool._pool_reference ,
                "pool_gap": pool._pool_gap,
                "individual_scores": pool._individual_score,
                "evaluation_results": pool._evaluation_results,
            }
            pool_dicts.append(pool_dict)

        return pool_dicts

    def check_evaluated(self):
        return all(p.get_score() is not None for p in self.pools)

    #########################################################
    # Evaluation and Update Pools
    #########################################################

    def extract_avg_value(self, data, key):
        """
        Safely extract the average value for a given key from a list of dicts.

        - Skips None entries and non-dict items.
        - Ignores missing or non-numeric values.
        - Returns 0.0 if no valid values found.
        """
        if not data:
            return 100_000_000

        key_values = []
        for d in data:
            if isinstance(d, dict):
                val = d.get(key)
                if isinstance(val, (int, float)):  # only count numeric values
                    key_values.append(val)

        if not key_values:
            return 0.0

        return sum(key_values) / len(key_values)

    def extract_avg_destroy_weights(self, data):
        """
        Compute the average destroy operator weights across all seeds for one pool's evaluation results.
        Returns a list of averages aligned with pool programs.
        """
        if not data:
            return [0.0] * POOL_SIZE

        # Assume all seeds have the same operator order
        num_ops = len(data[0]["results"].get("destroy_operator_weights", []))
        sums = [0.0] * num_ops
        counts = [0] * num_ops

        for d in data:
            results = d.get("results", {})
            weights = results.get("destroy_operator_weights", [])
            for i, w in enumerate(weights):
                sums[i] += w
                counts[i] += 1

        # Average per operator
        averages = [s / c if c > 0 else 0.0 for s, c in zip(sums, counts)]
        return averages

    def evaluate_and_update_pools(self, evaluation_results):

        print(len(evaluation_results))
        print(len(self.pools))

        for i, res in enumerate(evaluation_results):
            if res is None:
                self.pools[i].set_pool_score(100_000_000)
                continue


            if self.pools[i].get_score() is not None:
                continue

            avg_fitness = self.extract_avg_value(res, "fitness")
            avg_reference = self.extract_avg_value(res, "reference")
            avg_gap = self.extract_avg_value(res, "gap")

            print(avg_fitness, avg_reference, avg_gap)

            self.pools[i].set_pool_score(avg_fitness)
            self.pools[i].set_pool_reference(avg_reference)
            self.pools[i].set_pool_gap(avg_gap)

            avg_destroy_weights = self.extract_avg_destroy_weights(res)
            self.set_individual_scores(i, avg_destroy_weights)


    def mutate_pool(self, pop):
        """
        Mutate the best pools (top-ranked).
        Each child pool inherits some programs from the parent
        and fills the rest with fresh sampled programs.
        """

        if not self.check_evaluated():
            raise ValueError("[Pooling]: Cannot mutate pools before evaluation!")

        programs_all = self.db.load_programs_only(pop)
        new_pools = []

        # Get indices of the best pools
        best_indices = self.get_pool_best_pools(NR_OF_POOL_MUTATIONS)

        for idx in best_indices:
            parent = self.pools[idx]

            inherit_count = POOL_SIZE - NR_OF_MUTATIONS
            inherited = random.sample(parent.get_programs(), inherit_count)

            # sample the rest from the database
            sampled = random.sample(programs_all, NR_OF_MUTATIONS)

            # create child pool
            child_programs = inherited + sampled
            new_pool = Pool(child_programs)
            new_pool.create_pool_id(pop, idx, "mutate")
            new_pools.append(new_pool)

        # extend with children
        self.pools.extend(new_pools)

    #########################################################
    # Evaluation and Update Programs
    #########################################################

    def assign_scores_to_population(self, population: List[dict[str, any]]) -> List[dict[str, any]]:
        """
        Assign average scores from pools to each program in the given population.
        The scores are stored in offspring["objective"], and the population
        is returned sorted by this objective.
        """
        # Compute average program scores across all pools
        program_scores = self.compute_average_program_scores()

        # Assign them as objectives
        for prog in population:
            oid = prog["offspring"]["offspring_id"]
            avg_score = program_scores.get(oid, None)
            prog["offspring"]["objective"] = avg_score

        # Sort programs by objective (higher=better; flip reverse if lower=better)
        scored_population = sorted(
            population,
            key=lambda p: (
                p.get("offspring", {}).get("objective") is None,
                p.get("offspring", {}).get("objective")
            ),
            reverse=True
        )

        return scored_population

    def compute_average_program_scores(self):
        """
        Compute the average individual score of each program across all pools
        it appears in.

        Returns:
            dict mapping offspring_id -> average_score
        """
        score_sum = defaultdict(float)
        count = defaultdict(int)

        for pool in self.pools:
            programs = pool.get_programs()
            scores = pool.get_program_scores()

            for prog, score in zip(programs, scores):
                if score is None:
                    continue
                oid = prog["offspring"]["offspring_id"]
                score_sum[oid] += score
                count[oid] += 1

        # Compute averages
        avg_scores = {
            oid: (score_sum[oid] / count[oid]) if count[oid] > 0 else None
            for oid in score_sum
        }
        return avg_scores

    ##########################################################
    # POOL MANAGEMENT
    ##########################################################

    def keep_best_pools(self):
        """
        Keep only the top-NR_OF_POOLS pools based on their scores.
        Removes all lower-ranked pools from self.pools.
        """

        # Ensure all pools are scored
        if not self.check_evaluated():
            raise ValueError("[Pooling]: Cannot filter pools before evaluation!")

        # Get indices of top NR_OF_POOLS pools (best = lowest score)
        best_indices = self.get_pool_best_pools(NR_OF_POOLS)

        # Filter only those pools
        self.pools = [self.pools[i] for i in best_indices]

        print(f"[Pooling]: Kept {len(self.pools)} best pools (IDs: {[p.get_pool_id() for p in self.pools]})")
