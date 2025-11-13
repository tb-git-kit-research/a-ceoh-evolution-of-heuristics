import os
import time
import subprocess
from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_TYPE = "A-CEOH"
EOH_PROBLEM = "multibay_reshuffle_astar" # "multibay_reshuffle_astar" or "puzzle_astar_edu"

if "reshuffle" in EOH_PROBLEM:
    eoh_experiment_file = "exp_bay5_wh_1_fill_0.6.json"
if "puzzle" in EOH_PROBLEM:
    eoh_experiment_file = "20x20_200_edu.json"

# Get required environment variables
INSTANCES_PATH = os.getenv("INSTANCES_PATH")
BASE_PATH = os.getenv("BASE_PATH")

# Check if required environment variables are set
if not INSTANCES_PATH:
    print("INSTANCES_PATH not set.")
    print("Please create a .env file and add this variable")
    print("For more information, please look into the readme")
    time.sleep(10)
    exit(1)

if not BASE_PATH:
    print("BASE_PATH for evaluation not set.")
    print("Please create a .env file and add this variable")
    print("For more information, please look into the readme")
    time.sleep(10)
    exit(1)

#####################################################################################
#################                 General Settings                  #################
#####################################################################################

os.environ["PYTHONUNBUFFERED"] = "False"
os.environ["LOGGING"] = "False"
os.environ["DETAILED_OUTPUT"] = "False"  # Show all code and explanations

#####################################################################################
#################              Build Process and Execute            #################
#####################################################################################

print("########################## OLLAMA HOSTNAME ##########################")

# Install dependencies
subprocess.run(["pip", "install", "."])

os.environ["LLM_LOCAL_URL"] = "YOUR LOCAL URL"
print(os.environ["LLM_LOCAL_URL"])

print("########################## LLM MODELS ##########################")

MODEL_NAME = 'gpt-4o-2024-08-06' #'gpt-5-nano-2025-08-07' # deepseek-chat "gemma3:12b" "gemma3:27b" 'llama3.1:70b', 'gemma2:27b', 'nemotron:latest', 'qwen2.5-coder:32b', 'gpt-4o-2024-08-06'
os.environ["MODEL_NAME"] = MODEL_NAME

os.environ["PYTHONUNBUFFERED"] = "False"
os.environ["TIMEOUT"] = "120"  # Timeout for every request thread in seconds
os.environ["LOGGING"] = "False"
os.environ["DETAILED_OUTPUT"] = "False"  # Show all code and explanations



if EXPERIMENT_TYPE == "EOH":
    USE_PROBLEM_CONTEXT = 0
    EOH_PROBLEM = EOH_PROBLEM.strip("_astar") + "_uninformed_astar"

if EXPERIMENT_TYPE == "P-CEOH":
    USE_PROBLEM_CONTEXT = 1
    EOH_PROBLEM = EOH_PROBLEM.strip("_astar") + "_uninformed_astar"

if EXPERIMENT_TYPE == "A-CEOH":
    USE_PROBLEM_CONTEXT = 0
    EOH_PROBLEM = EOH_PROBLEM

if EXPERIMENT_TYPE == "PA-CEOH":
    USE_PROBLEM_CONTEXT = 1
    EOH_PROBLEM = EOH_PROBLEM

print(f"RUNNING {EOH_PROBLEM}")
print(f"CONTEXT: {EXPERIMENT_TYPE}")

os.environ["EOH_PROBLEM"] = EOH_PROBLEM


N = 1
for i in range(N):  # Run the strategy N times
    subprocess.run([
                "ceoh", "run", EOH_PROBLEM,
                "--model_name", MODEL_NAME,
                "--ec_operators", "e1,e2,m1,m2",
                "--ec_n_pop", "20",
                "--ec_pop_size", "20",
                "--ec_m", "5",
                "--use_example", str(USE_PROBLEM_CONTEXT),
                "--eoh_experiment_file", eoh_experiment_file,
            ])

