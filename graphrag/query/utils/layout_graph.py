import os

import matplotlib.pyplot as plt
import networkx as nx
from load import load_parquet


def load_graph(data_dir: str):
    # Load data from parquet
    entities = load_parquet(os.path.join(data_dir, "entities.parquet"))
    relations = load_parquet(os.path.join(data_dir, "relations.parquet"))
    
    # Construct graph
    # Add node
    graph = nx.Graph()
    for entity in entities:
        graph.add_node(entity["id"], name=entity["name"])
    
    for entity in entities:
        if entity["alias"]:
            for alias in entity["alias"]:
                graph.add_edge(entity["id"], alias)
    
    # Add edge
    for relation in relations:
        graph.add_edge(relation["source"], relation["target"], description=relation["description"])
    
    return graph


def layout(data_dir: str):
    graph = load_graph(data_dir)
    node_labels = nx.get_node_attributes(graph, "name")
    edge_labels = nx.get_edge_attributes(graph, "description")
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        labels=node_labels,
        node_color="skyblue",
        node_size=200,
        font_size=8,
        font_color="black",
        edge_color="gray",
    )
    # 绘制边上的标签
    # nx.draw_networkx_edge_labels(
    #     graph,
    #     pos,
    #     edge_labels=edge_labels,
    #     font_color="red",
    #     font_size=4
    # )
    # 显示图
    plt.title("Entity-Relation Graph")
    plt.show()


def main() -> None:
    layout("D:/Project/E-GraphRAG/experiments/rag/output/3")


if __name__ == "__main__":
    main()