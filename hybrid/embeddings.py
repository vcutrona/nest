def etp(graph, v1, v2, denominator):
    return graph.get_edge_data(v1, v2)['weight'] / denominator


def convert(uri):
    return "<" + uri + ">"


def thinOut(graph):
    count = int(0.75 * len(graph.edges.data("weight")))
    return sorted(graph.edges.data("weight"), key=lambda tup: tup[2])[:count]
