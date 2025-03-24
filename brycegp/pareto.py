from sklearn.metrics import mean_squared_error

def pareto_front(archive, archive_size, candidates, X, y):
    def dominates(individual1, individual2):
        error1, complexity1 = mean_squared_error(individual1(X), y), individual1.complexity()
        error2, complexity2 = mean_squared_error(individual2(X), y), individual2.complexity()
        all_better_or_equal = (error1 <= error2 and complexity1 <= complexity2)
        strictly_better = (error1 < error2 or complexity1 < complexity2)
        return all_better_or_equal and strictly_better

    layers = []

    while candidates:
        current_layer = []
        for candidate in candidates:
            is_dominated = any(dominates(other, candidate) for other in candidates if str(other) != str(candidate))
            if not is_dominated:
                current_layer.append(candidate)
        layers.append(current_layer)
        candidates = [candidate for candidate in candidates if str(candidate) not in [str(x) for x in current_layer]]

    for layer in layers:
        archive.extend(layer)
        if len(archive) >= archive_size:
            break