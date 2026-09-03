
def check_interaction(graph) -> bool:
    edge_type = graph.edge_type

    if not 2 in edge_type:
        return False

    return True
