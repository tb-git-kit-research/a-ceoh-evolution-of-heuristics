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
