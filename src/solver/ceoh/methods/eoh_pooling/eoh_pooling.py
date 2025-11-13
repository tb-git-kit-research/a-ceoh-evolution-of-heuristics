import random
import time

from .eoh_pooling_interface import PoolingInterface
from .database import PopulationDatabase
from solver.ceoh.utils.visualizeResults import *
from solver.ceoh.utils.visualizePromptFrequency import *

from .pooling_class import CeohPooling
from .util import get_offsprings

DEBUG_MODE = False

class EOH_POOLING:
    def __init__(self, paras, problem, select, manage, **kwargs):
        self.paras = paras
        self.prob = problem
        self.select = select

        self.operators = paras.ec_operators

        self.n_pop = 20

        self.manage = manage
        self.pop_size = paras.ec_pop_size

        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2

        self.output_path = paras.exp_output_path
        try:
            self.result_folder_name = os.environ["CURRENT_EXPERIMENT"]
        except:
            self.result_folder_name = None

        # TODO: Remove after verify functions
        if DEBUG_MODE:
            self.result_folder_name = os.path.join(os.environ["OUTPUT_PATH"], "results_20251005_200415_lns_cvrp_pool_gpt5nano20250807_ueFalse_uiFalse_uitypeNone")

        # database handler
        self.db = PopulationDatabase(self.result_folder_name)

        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_example = paras.ec_use_example

        random.seed(2024)


    def add2pop(self, population, offspring):
        for off in offspring:
            population.append(off)


    def _get_id_name(self, op, pop):
        time_str = datetime.now().strftime("%y%m%d_%H%M%S")
        return f"pool_op_{op}_pop_{pop}_{time_str}"


    def run(self):
        print("- Evolution Start -")
        time_start = time.time()

        interface_ec = PoolingInterface(self.paras, self.prob, self.select)

        if not DEBUG_MODE:

            # Create initia
            print("creating initial population:")
            current_pop = []
            for i in range(20):
                file = self._get_id_name("i1", 0)
                current_file = interface_ec.run_heuristic_generation([], "i1", file)
                self.db.save_program(current_file, file, current_file["exception"])
                current_pop.append(current_file)


            print("initial population has been created!")
            self.db.save_population(current_pop, generation=0, is_best=False)
        else:
            current_pop = self.db.load_programs_only(0)
            self.db.save_population(current_pop, generation=0, is_best=False)

        print("Start initial Pooling")
        pooling = CeohPooling(self.db)

        pooling.add_new_pools(0)

        _, results = interface_ec.run_evaluation(pooling.pools)

        print("Evaluate Pools")
        pooling.evaluate_and_update_pools(results)

        print("Save Pools")
        pool_dict = pooling.convert_to_dict()
        self.db.save_pools(pool_dict, 0)
        self.db.save_pools(pool_dict, 0,True)

        print("Evaluate Programs")
        population = pooling.get_programs_from_best_pools()
        print(population)
        scored_population = pooling.assign_scores_to_population(population)

        print(f"Save Programs")
        self.db.save_population(scored_population, generation=0, is_best=False)
        self.db.save_population(scored_population, generation=0, is_best=True)

        n_start = 1

        # Main loop
        for pop in range(n_start, n_start + self.n_pop, 1):
            print(f"Start with Population: {pop}!")

            current_population = []
            last_pop = self.db.load_population(pop-1)
            last_offsprings = get_offsprings(last_pop)

            for op in self.operators:
                for i in range(4):
                    file = self._get_id_name(op, pop)
                    current_file = interface_ec.run_heuristic_generation(last_offsprings, op, file)
                    self.db.save_program(current_file, file, current_file["exception"])
                    current_population.append(current_file)


            print(f"Mutate Pool of Population {pop}!")
            pooling.mutate_pool(pop)

            print(f"Add new Pools of Population {pop}!")
            pooling.add_new_pools(pop)

            print(f"Evaluation of Pools in Population {pop}!")
            parents, results = interface_ec.run_evaluation(pooling.pools)

            print("Save Pools")
            pool_dict = pooling.convert_to_dict()
            self.db.save_pools(pool_dict, pop)

            print(f"Update Pool Performance Metric of Population{pop}!")
            pooling.evaluate_and_update_pools(results)

            print("Keeping only the best pools...")
            pooling.keep_best_pools()

            print(f"Save pool and results of Population {pop}!")
            pool_dict = pooling.convert_to_dict()
            self.db.save_pools(pool_dict, pop)
            self.db.save_pools(pool_dict, pop, only_best=True)

            print(f"Population Management of Population {pop}!")
            population = pooling.get_programs_from_best_pools()

            # Compute average program scores across all pools
            scored_population = pooling.assign_scores_to_population(population)

            print(f"Save scored Population {pop}!")
            self.db.save_population(scored_population, generation=pop, is_best=False)
            self.db.save_population(scored_population, generation=pop, is_best=True)

            #TODO POOL POPULATION MANAGEMENT

            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")

            try:
                load_and_plot_objective_ranges(self.result_folder_name)
            except Exception as e:
                print(f"Error plotting objective ranges: {e}")
            try:
                plot_evaluation_time(self.result_folder_name)
            except Exception as e:
                print(f"Error plotting evaluation time: {e}")