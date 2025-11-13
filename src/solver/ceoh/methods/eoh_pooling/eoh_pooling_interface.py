# Copyright (c) 2024 - Fei Liu (fliu36-c@my.cityu.edu.hk)
#
# Modified by: 
#           Thomas Bömer (thomas.bömer@tu-dortmund.de)
#           Nico Koltermann (nico.koltermann@tu-dortmund.de) 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from copy import deepcopy

import numpy as np
import time
import traceback
from solver.ceoh.methods.eoh.eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
import concurrent.futures

import os


import multiprocessing
import logging


class PoolingInterface():
    def __init__(self, paras, interface_prob, select, **kwargs):

        self.detailed_output = os.environ["DETAILED_OUTPUT"] == 'True'

        print(f'Config timout is ignored: {paras.eva_timeout}')

        # LLM settings
        self.pop_size = paras.ec_pop_size
        self.n_p = paras.exp_n_proc
        self.select = select
        self.m = paras.ec_m

        self.interface_eval = interface_prob

        self.evol = Evolution(paras, interface_prob.prompts, **kwargs)

        warnings.filterwarnings("ignore")


    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def population_generation(self):
        
        n_create = 2
        population = []

        for i in range(n_create):
            _,pop = self.init_evaluation([],'i1')
            for p in pop:
                population.append(p)
        return population
    
    def population_generation_seed(self,seeds,n_p):

        population = []
        fitness = []

        if self.n_p > 1:
            fitness, detailed_scores = Parallel(n_jobs=self.n_p)(
                delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)

        else:
            for seed in seeds:
                f, _ = self.interface_eval.evaluate(seed['code'])
                fitness.append(f)

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = float(np.round(obj, 5))
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population
    
    def _get_alg(self,pop,operator):

        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }

        if operator == "i1":
            parents = None
            [offspring['code'],offspring['algorithm']], prompt, full_res =  self.evol.i1()
        elif operator == "e1":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.e1(parents)
        elif operator == "e2":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.e2(parents)
        elif operator == "m1":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.m1(parents[0])
        elif operator == "m2":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.m2(parents[0])
        elif operator == "m3":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.m3(parents[0])
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")

        return parents, offspring, prompt, full_res

    def run_heuristic_generation(self, population, op, filename=""):

        start_time = time.perf_counter_ns()
        possible_exception = ""

        try:
            parents, offspring, prompt, full_res = self._get_alg(population, op)
        except Exception as e:
            traceback.print_exc()

            prompt = None
            parents = None
            full_res = None
            offspring = { "code": None, "objective": None }
            possible_exception = e

        # Test if the program is executable and valid
        try:
            offspring["offspring_id"] = "check_only"
            check_input = [{
                "offspring": offspring,
            }]
            avg_fitness, results = self.interface_eval.evaluate(check_input, check_only = True)

        except Exception as e:
            possible_exception = e

        offspring_save = deepcopy(offspring)

        end_time = time.perf_counter_ns()
        elapsed_time = (end_time - start_time)

        offspring_save['evaluation_time'] = elapsed_time
        offspring_save['offspring_id'] = filename

        current_file = {
            'exception': str(possible_exception),
            'p': parents,
            'offspring': offspring_save,
            'original_code': offspring_save["code"],
            'prompt': prompt,
            'full_request': full_res,
            'details': None,
        }

        return current_file

    def _generate_heuristic(self, nr_to_generate, pop, operator):
        programs = []
        for i in range(nr_to_generate):
            programs.append(self.get_algorithm(pop, operator))
        return programs

    def _eval_heuristic(self, pool):

        start_time = time.perf_counter_ns()

        if self.n_p > 1:

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.interface_eval.evaluate, pool)
                fitness, detailed_scores = future.result(timeout=self.timeout)
                future.cancel()
        else:
            fitness, detailed_scores = self.interface_eval.evaluate(pool)

        end_time = time.perf_counter_ns()
        elapsed_time = (end_time - start_time)

        print(f"Thread Time: {elapsed_time}")
        logging.info(f"Thread Time: {elapsed_time}")

        return fitness, detailed_scores

    def run_evaluation(self, pools):
        results = [None] * len(pools)

        try:
            unevaluated = [(i, pool) for i, pool in enumerate(pools) if pool.get_score() is None]

            if unevaluated:
                eval_results = Parallel(n_jobs=self.n_p)(
                    delayed(self._eval_heuristic)(pool.get_programs())
                    for _, pool in unevaluated
                )

                for (i, pool), res in zip(unevaluated, eval_results):
                    results[i] = res
                    pool.set_evaluation_results(res)  # <--- store raw results

            for i, pool in enumerate(pools):
                if results[i] is None and pool.get_evaluation_results() is not None:
                    results[i] = pool.get_evaluation_results()

        except multiprocessing.TimeoutError:
            logging.error("TIMEOUT ERROR IN PARALLEL: This should not be possible!.")
            print("TIMEOUT ERROR IN PARALLEL: This should not be possible!.")
        except Exception as e:
            logging.error(f"Error: {e}")
            print(traceback.format_exc())
            print(f"Error - PoolingInterface: {e}")

        time.sleep(0.5)

        print(results)

        out_p, out_off = [], []
        for p, off in results:
            out_p.append(p)
            out_off.append(off)

        return out_p, out_off

