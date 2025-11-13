# Copyright (c) 2025
#           (adapted to align with MULTIBAY_RESHUFFLECONST_ASTAR structure)
#
# MIT License (same as your original)

import os
import time
import types
import itertools
import threading
import heapq
from copy import deepcopy
from multiprocessing import Process, Queue
from dotenv import load_dotenv

from solver.ceoh.problems.descriptions.puzzle.puzzle_astar_edu.puzzle_generator import load_instances_from_file
from solver.ceoh.utils.getParas import Paras
from solver.ceoh.problems.descriptions.puzzle.puzzle_astar_edu.prompts import GetPrompts

# ---------------------------------------------------------------------
# Constants (these may be overridden via paras)
# ---------------------------------------------------------------------
TIMEOUT_SECONDS = 60
NUM_EVAL_INSTANCES = 10
MAX_MOVES = 200
PRINTOUT = False
N_WORKERS = 10
MAX_EVALUATED_NODES = 1_000_000  # extra guard, similar to warehouse script

# ---------------------------------------------------------------------
# A* Node
# ---------------------------------------------------------------------
class PuzzleNode:
    def __init__(self, puzzle, g, heuristic_fn, parent=None, move=None):
        self.puzzle = puzzle
        self.N = len(puzzle)
        self.g = g
        # heuristic_fn expects a *state*, pass a deep copy to ensure isolation
        self.h = heuristic_fn(deepcopy(puzzle))
        self.f = self.g + self.h
        self.parent = parent
        self.move = move
        self._ser = None  # cache serialize

    def __lt__(self, other):
        # tie-break by f then g for determinism if equal f (implicit with heap key as well)
        return (self.f, self.g) < (other.f, other.g)

    def find_blank(self):
        for r in range(self.N):
            for c in range(self.N):
                if self.puzzle[r][c] == 0:
                    return r, c

    def get_neighbors(self, heuristic_fn):
        neighbors = []
        r, c = self.find_blank()
        directions = {(-1, 0): 'U', (1, 0): 'D', (0, -1): 'L', (0, 1): 'R'}
        for (dr, dc), move in directions.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.N and 0 <= nc < self.N:
                new_puzzle = deepcopy(self.puzzle)
                new_puzzle[r][c], new_puzzle[nr][nc] = new_puzzle[nr][nc], new_puzzle[r][c]
                neighbors.append(PuzzleNode(new_puzzle, self.g + 1, heuristic_fn, parent=self, move=move))
        return neighbors

    def is_goal(self):
        flat = [num for row in self.puzzle for num in row]
        return flat == list(range(1, self.N * self.N)) + [0]

    def get_objective_value(self):
        if not self.is_goal():
            return MAX_MOVES
        return len(self.reconstruct_path(self))

    @staticmethod
    def reconstruct_path(node):
        path = []
        while node.parent is not None:
            path.append((node.move, node.puzzle))
            node = node.parent
        return list(reversed(path))

    def serialize(self):
        if self._ser is None:
            self._ser = tuple(tuple(row) for row in self.puzzle)
        return self._ser

    # Optional visual debug
    def print_highlighted_path(path_nodes):
        def print_puzzle_with_highlight(old_state, new_state):
            size = len(new_state)
            for r in range(size):
                row_str = ""
                for c in range(size):
                    val = new_state[r][c]
                    display = "  " if val == 0 else f"{val:2}"
                    if old_state is not None and old_state[r][c] != new_state[r][c]:
                        row_str += f"*{display}* "
                    else:
                        row_str += f" {display}  "
                print(row_str)
            print()

        print("Initial Puzzle State:")
        print_puzzle_with_highlight(None, path_nodes[0].puzzle)

        for i in range(1, len(path_nodes)):
            print(f"Move {i}: {path_nodes[i].move}")
            print_puzzle_with_highlight(path_nodes[i - 1].puzzle, path_nodes[i].puzzle)


# ---------------------------------------------------------------------
# Core A* (returns a dict mirroring your warehouse A* result shape)
# ---------------------------------------------------------------------
def astar_puzzle_core(heuristics, start_puzzle):
    open_list = []
    visited = set()
    evaluated_nodes = 0
    counter = itertools.count()

    root = PuzzleNode(start_puzzle, g=0, heuristic_fn=heuristics.score_state)
    heapq.heappush(open_list, (root.f, next(counter), root))
    visited.add(root.serialize())
    start = time.monotonic()
    last_popped = root

    while open_list:
        # Timeout / node cap check first (consistent, but last_popped is defined)
        if (time.monotonic() - start) > TIMEOUT_SECONDS or evaluated_nodes > MAX_EVALUATED_NODES:
            return {
                'g_score': None,
                'h_score': None,
                'f_score': None,
                'evaluated_nodes': evaluated_nodes,
                "solution_time": time.monotonic() - start,
                "solution": None,
                "number_moves": MAX_MOVES,
                "loaded_move_distance": None,    # not applicable
                "unloaded_move_distance": None,  # not applicable
                "objective_value": MAX_MOVES
            }

        _, _, current = heapq.heappop(open_list)
        last_popped = current

        if current.is_goal():
            path = current.reconstruct_path(current)
            if PRINTOUT:
                # Build node chain for pretty printer
                chain = []
                n = current
                while n is not None:
                    chain.append(n)
                    n = n.parent
                chain.reverse()
                current.print_highlighted_path(chain)

            return {
                'g_score': current.g,
                'h_score': current.h,
                'f_score': current.f,
                'evaluated_nodes': evaluated_nodes,
                "solution_time": time.monotonic() - start,
                "solution": [move for move, _ in path],  # sequence of 'UDLR'
                "number_moves": len(path),
                "loaded_move_distance": None,
                "unloaded_move_distance": None,
                "objective_value": current.get_objective_value()
            }

        for neighbor in current.get_neighbors(heuristics.score_state):
            state = neighbor.serialize()
            if state in visited:
                continue
            evaluated_nodes += 1
            visited.add(state)
            # Cheap tie-breaker with counter to keep determinism
            heapq.heappush(open_list, (neighbor.f, next(counter), neighbor))

    # Exhausted without a goal
    return {
        'g_score': None,
        'h_score': None,
        'f_score': None,
        'evaluated_nodes': evaluated_nodes,
        "solution_time": time.monotonic() - start,
        "solution": None,
        "number_moves": MAX_MOVES,
        "loaded_move_distance": None,
        "unloaded_move_distance": None,
        "objective_value": MAX_MOVES
    }


# ---------------------------------------------------------------------
# Multiprocessing worker (loads heuristic ONCE, like your warehouse worker)
# ---------------------------------------------------------------------
def puzzle_worker_main(task_queue, result_queue, code_string):
    # Load heuristic once per process
    heuristic_module = types.ModuleType("heuristic_module")
    exec(code_string, heuristic_module.__dict__)

    while True:
        job = task_queue.get()
        if job is None:
            break

        try:
            config = job  # here: a dict containing 'puzzle' and 'lb'
            puzzle = config['puzzle']
            lb = config.get('lb', 1)

            results = astar_puzzle_core(heuristic_module, puzzle)

            # Fitness: relative gap vs. lower bound (like your MR version used a baseline)
            # Guard lb == 0
            g = results['number_moves'] if results['number_moves'] is not None else MAX_MOVES
            current_score = (g - lb) / lb if lb != 0 else float(g)

            result_queue.put({
                'fitness': current_score,
                'reference': lb,
                'results': results
            })
        except Exception as e:
            result_queue.put({'fitness': None, 'error': str(e)})


# ---------------------------------------------------------------------
# Manager / Evaluator class (mirrors MULTIBAY_RESHUFFLECONST_ASTAR)
# ---------------------------------------------------------------------
class PUZZLE_ASTAR:
    def __init__(self, instance_file=None, code_string=None, paras=None, seed = None):
        """
        paras may override:
         - TIMEOUT_SECONDS
         - NUM_EVAL_INSTANCES
         - MAX_MOVES
         - N_WORKERS
        and provide eoh_experiment_file
        """
        global TIMEOUT_SECONDS, NUM_EVAL_INSTANCES, MAX_MOVES, N_WORKERS, MAX_EVALUATED_NODES

        self.prompts = GetPrompts()
        self.paras = paras
        self.code_string = code_string

        if paras is not None:
            print(paras)
            # Pull overrides if present
            TIMEOUT_SECONDS = getattr(paras, 'TIMEOUT_SECONDS', TIMEOUT_SECONDS)
            print(TIMEOUT_SECONDS)
            NUM_EVAL_INSTANCES = getattr(paras, 'NUM_EVAL_INSTANCES', NUM_EVAL_INSTANCES)
            MAX_MOVES = getattr(paras, 'MAX_MOVES', MAX_MOVES)
            N_WORKERS = getattr(paras, 'N_WORKERS', N_WORKERS)
            MAX_EVALUATED_NODES = getattr(paras, 'MAX_EVALUATED_NODES', MAX_EVALUATED_NODES)

        self.instance_file = paras.eoh_experiment_file if paras and hasattr(paras, 'eoh_experiment_file') else instance_file
        all_instances = load_instances_from_file(self.instance_file)
        all_instances = all_instances[:NUM_EVAL_INSTANCES]

        if seed:
            all_instances = [all_instances[seed]]

        # Prepare configs per instance (mirrors MR style of passing config)
        self.instance_configs = []
        for sample in all_instances:
            self.instance_configs.append({
                'puzzle': sample['puzzle'],
                'lb': sample.get('misplaced', 1)
            })

        if len(self.instance_configs) == 0:
            print("[INSTANCES ERROR]: No puzzle instances available")
            return

        if self.code_string is not None:
            self.fitness, self.details = self.evaluate(self.code_string, self.paras)

    def evaluate(self, code_string, paras=None):
        try:
            number_of_exp = len(self.instance_configs)

            managers = []
            for _ in range(N_WORKERS):
                task_queue = Queue()
                result_queue = Queue()
                p = Process(target=puzzle_worker_main, args=(task_queue, result_queue, code_string))
                p.start()
                managers.append((p, task_queue, result_queue))

            results = [None] * number_of_exp
            buckets = [[] for _ in range(N_WORKERS)]
            for i, config in enumerate(self.instance_configs):
                buckets[i % N_WORKERS].append((i, config))

            threads = []

            def run_manager(index, jobs):
                process, task_queue, result_queue = managers[index]
                for i, config in jobs:
                    task_queue.put(config)
                    start = time.monotonic()
                    result = None
                    # Allow worker to exceed TIMEOUT slightly to flush queue
                    while (time.monotonic() - start) < (TIMEOUT_SECONDS + 5):
                        try:
                            candidate = result_queue.get(timeout=0.1)
                            if candidate is not None:
                                result = candidate
                                break
                        except Exception:
                            pass
                    # Fill slot
                    results[i] = result
                task_queue.put(None)

                # Wait a little for graceful shutdown
                process.join(timeout=0.1)

                # If still alive (e.g., heuristic stuck in infinite loop), kill it
                if process.is_alive():
                    print(f"[WARN] Worker {index} did not exit, force killing.")
                    process.terminate()
                    process.join()

            for i in range(N_WORKERS):
                t = threading.Thread(target=run_manager, args=(i, buckets[i]))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            # Aggregate like MR version
            detailed_fitness = []
            references = []
            algo_moves = []
            solution_times = []
            evaluated_nodes = []
            solutions = []

            overall_score = 0.0

            for idx, result in enumerate(results):
                lb = self.instance_configs[idx]['lb']

                # Robust fallback if worker failed or timed out
                if (result is None) or (result.get('fitness') is None):
                    # synthesize a failure record
                    fit = float(MAX_MOVES) if lb == 0 else (MAX_MOVES - lb) / lb
                    detailed_fitness.append(fit)
                    references.append(lb)
                    algo_moves.append(MAX_MOVES)
                    solutions.append(None)
                    solution_times.append(TIMEOUT_SECONDS + 5)
                    evaluated_nodes.append(0)
                    overall_score += fit
                    print(f"[WARN] Worker failed on instance {idx}. Using fallback gap={fit:.3f}.")
                    continue

                detailed_fitness.append(result['fitness'])
                references.append(result['reference'])

                res = result['results']
                algo_moves.append(res['number_moves'])
                solutions.append(res['solution'])
                solution_times.append(res['solution_time'])
                evaluated_nodes.append(res['evaluated_nodes'])

                overall_score += result['fitness']

            fitness = overall_score / number_of_exp
            details = {
                'detailed_fitness': detailed_fitness,
                'references': references,
                "move_numbers": algo_moves,
                "solutions": solutions,
                "solution_time": solution_times,
                "evaluated_nodes": evaluated_nodes,
            }
            print(details)
            print(f'Overall fitness (avg gap): {fitness}', flush=True)
            return fitness, details

        except Exception as e:
            print(f"[ERROR] Exception in evaluate: {e}")
            return None, None


# ---------------------------------------------------------------------
# Run Example
# ---------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()

    code_string = """
def score_state(state):
    N = len(state)
    score = 0
    target_positions = {num: divmod(num - 1, N) for num in range(1, N * N)}
    
    for i in range(N):
        for j in range(N):
            tile = state[i][j]
            if tile != 0:
                target_i, target_j = target_positions[tile]
                manhattan_distance = abs(i - target_i) + abs(j - target_j)
                score += manhattan_distance
                
                # Check horizontal blocking tiles
                for k in range(min(j, target_j), max(j, target_j) + 1):
                    if k != j and state[i][k] != 0:
                        other_target_i, other_target_j = target_positions[state[i][k]]
                        if (other_target_i == i and 
                            ((target_j < j and other_target_j > target_j) or 
                             (target_j > j and other_target_j < target_j))):
                            blocking_distance = abs(other_target_j - target_j)
                            score += blocking_distance * (1 + (N - min(target_j, other_target_j)) / N)
                
                # Check vertical blocking tiles
                for k in range(min(i, target_i), max(i, target_i) + 1):
                    if k != i and state[k][j] != 0:
                        other_target_i, other_target_j = target_positions[state[k][j]]
                        if (other_target_j == j and 
                            ((target_i < i and other_target_i > target_i) or 
                             (target_i > i and other_target_i < target_i))):
                            blocking_distance = abs(other_target_i - target_i)
                            score += blocking_distance * (1 + (N - min(target_i, other_target_i)) / N)
    
    return score

"""

    paras_instance = Paras()
    paras_instance.set_paras(
        eoh_experiment_file="20x20_200_edu.json",
    )

    puzzle_solver = PUZZLE_ASTAR(code_string=code_string, paras=paras_instance, seed = 0)
    print("Final Fitness:", getattr(puzzle_solver, 'fitness', None))
    print("Details Dict:", getattr(puzzle_solver, 'details', None))
