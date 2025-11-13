
def get_destroy_scores(data):
    try:
        weights = data["results"]["destroy_operator_weights"]
    except:
        weights = None

    return weights

def get_destroy_counts(data):
    return data["results"]["destroy_operator_counts"]

def get_results_fitness(data):
    return data["fitness"]

def get_agv_weight(data):
    try:
        weights = data["results"]["destroy_operator_weights"]
    except:
        return None

    return sum(weights) / len(weights)


def get_offsprings(data: list[dict]) -> list[str]:
    return [ind['offspring'] for ind in data]