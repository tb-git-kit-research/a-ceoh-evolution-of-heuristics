

class Probs():
    def __init__(self,paras):

        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "multibay_reshuffle_astar":
            from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar import run
            self.prob = run.MULTIBAY_RESHUFFLECONST_ASTAR(paras.eoh_experiment_file)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "multibay_reshuffle_uninformed_astar":
            from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_uninformed_astar import run
            self.prob = run.MULTIBAY_RESHUFFLECONST_ASTAR(paras.eoh_experiment_file)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "puzzle_astar_edu":
            from solver.ceoh.problems.descriptions.puzzle.puzzle_astar_edu import run
            self.prob = run.PUZZLE_ASTAR(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "puzzle_astar_edu_uninformed_astar":
            from solver.ceoh.problems.descriptions.puzzle.puzzle_astar_edu_uninformed_astar import run
            self.prob = run.PUZZLE_ASTAR(code_string=None, paras=paras)
            print("- Prob " + paras.problem + " loaded ")

        else:
            print("problem "+paras.problem+" not found!")


    def get_problem(self):

        return self.prob
