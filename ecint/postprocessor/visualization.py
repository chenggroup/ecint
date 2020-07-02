from aiida.orm import load_node
from aiida.tools.visualization import Graph


def get_provenance_graph(pk, level, annotate_links=None, graph_attr=None):
    """
    generate aiida provenance graph for a node
    :param pk: node pk
    :param level: 'minimal', 'low', 'medium', 'high'
    :param annotate_links: 'label', 'type', 'both', description of edges of provenance graph
    :param graph_attr: attribute for graphviz, more information: https://www.graphviz.org/doc/info/attrs.html
    :return:
    """
    if graph_attr is None:
        graph_attr = {"rankdir": "TR"}
    graph = Graph(graph_attr=graph_attr)
    if level == 'minimal':
        called_list = [node.pk for node in load_node(pk).called]
        graph.add_incoming(pk)
        graph.add_outgoing(pk)
        for called_pk in called_list:
            graph.add_incoming(called_pk)
            graph.add_outgoing(called_pk, link_types='return')
    elif level == 'low':
        if annotate_links is None:
            annotate_links = 'label'
        called_list = [node.pk for node in load_node(pk).called]
        graph.add_incoming(pk, annotate_links=annotate_links)
        graph.add_outgoing(pk, annotate_links=annotate_links)
        for called_pk in called_list:
            graph.add_incoming(called_pk, annotate_links=annotate_links)
            graph.add_outgoing(called_pk, annotate_links=annotate_links, link_types='return')
    elif level == 'medium':
        if annotate_links is None:
            annotate_links = 'label'
        singleworkchain_list = [node.pk for node in load_node(pk).called]
        baseworkchain_list = [node.pk for node in [load_node(pk).called[0] for pk in singleworkchain_list]]
        calculation_list = [node.pk for node in [load_node(pk).called[0] for pk in baseworkchain_list]]
        code_pk = load_node(calculation_list[0]).get_incoming().get_node_by_label('code')

        graph.add_incoming(pk, annotate_links=annotate_links)
        graph.add_outgoing(pk, annotate_links=annotate_links)
        for node_pk in singleworkchain_list:
            graph.add_incoming(node_pk, annotate_links=annotate_links)
            graph.add_outgoing(node_pk, annotate_links=annotate_links)

        for node_pk in baseworkchain_list:
            graph.add_incoming(node_pk, annotate_links=annotate_links, link_types='call_work')
            graph.add_outgoing(node_pk, annotate_links=annotate_links, link_types='call_calc')

        for node_pk in calculation_list:
            graph.add_incoming(node_pk, annotate_links=annotate_links, link_types='call_calc')
            graph.add_outgoing(node_pk, annotate_links=annotate_links, link_types='return')

        graph.add_node(code_pk)
        for node_pk in calculation_list:
            graph.add_edge(code_pk, node_pk)
    elif level == 'high':
        if annotate_links is None:
            annotate_links = 'both'
        graph.recurse_descendants(
            pk,
            include_process_inputs=True,
            annotate_links=annotate_links,
        )
    graph.graphviz.render(f'provenance_graph_{level}', format='png')
