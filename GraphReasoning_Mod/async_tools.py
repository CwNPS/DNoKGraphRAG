

from GraphReasoning_Mod.graph_tools import *
from GraphReasoning_Mod.utils import *
from GraphReasoning_Mod.graph_generation import *
from concurrent.futures import ThreadPoolExecutor
import asyncio
from queue import Queue
from threading import Thread, Lock
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_generate_graph_triplets(inputs, llm, metadata):
    try:
        SYS_PROMPT_GRAPHMAKER = """
        You are a specialized system for extracting relationship networks from mathematical texts. 
        Your task is to identify key concepts and their specific relationships from the given context (delimited by ```). 
        Output in JSON format.
        """
        USER_PROMPT = f"Context: ```{inputs}```\n\nOutput:"
        response = await asyncio.to_thread(llm.invoke, SYS_PROMPT_GRAPHMAKER + "\n" + USER_PROMPT)

        response_json = json.loads(response[response.find("["):response.rfind("]") + 1])
        for item in response_json:
            item.update(metadata)
        return response_json
    except Exception as e:
        traceback.print_exc()
        return None

async def process_batches_async(dataframe, llm, batch_size=10):
    results = []
    for i in tqdm(range(0, len(dataframe), batch_size), desc="Processing Batches"):
        batch = dataframe.iloc[i:i + batch_size]
        tasks = [
            async_generate_graph_triplets(row.text, llm, {
                "chunk_id": row.chunk_id,
                "source": row.source,
                "metadata": row.metadata
            })
            for _, row in batch.iterrows()
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend([res for res in batch_results if res])
    return results

async def make_graph_from_docs_async(docs, llm, **kwargs):

    chunks = split_text_for_kg(docs, kwargs.get("chunk_size", 2500), kwargs.get("chunk_overlap", 500))
    df = documents2Dataframe(chunks)

    concepts_list = await process_batches_async(df, llm)

    dfg1 = pd.DataFrame(concepts_list)
    dfg1['metadata'] = dfg1['metadata'].apply(lambda x: str(x))

    if kwargs.get("add_source_nodes", True):
        dfg1 = add_source_relationship(dfg1)

    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1['count'] = 4

    if kwargs.get("include_contextual_proximity", True):
        dfg2 = contextual_proximity(dfg1)
        dfg = pd.concat([dfg1, dfg2], axis=0)
    else:
        dfg = dfg1

    dfg['metadata'] = dfg['metadata'].apply(lambda x: str(x))

    dfg = (
        dfg.groupby(["node_1", "node_2"])
        .agg({
            "chunk_id": ",".join,
            "edge": 'first',
            'count': 'sum',
            'metadata': 'first'
        })
        .reset_index()
    )

    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
    G = nx.Graph()

    for node in nodes:
        G.add_node(str(node))

    for _, row in dfg.iterrows():
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            metadata=row["metadata"],
            weight=row['count'] / 4
        )

    graph_GraphML = f'{kwargs.get("data_dir", "./data")}/{kwargs.get("graph_root", "graph")}_graphML.graphml'
    await asyncio.to_thread(nx.write_graphml, G, graph_GraphML)

    return graph_GraphML, G

async def graph_generator(batch_docs, llm, graph_queue):
    """Generate graphs and add them to a thread safe queue."""
    graph_root = f'batch_{uuid.uuid4().hex[:8]}'
    _, batch_graph = await make_graph_from_docs_async(
        docs=batch_docs,
        llm=llm,
        include_contextual_proximity=False,
        graph_root=graph_root,
        chunk_size=2500,
        chunk_overlap=0,
        repeat_refine=0,
        verbatim=True,
        data_dir='./data/batch_graphs/'
    )
    await graph_queue.put(batch_graph)

async def graph_processor(graph_queue: Queue, final_graph, embd_object, lock, exception_handler, **kwargs):
    """Process graphs from the queue and perform simplification."""
    while True:
        try:
            batch_graph = await asyncio.wait_for(graph_queue.get(), timeout=1800) # 30 minutes timeout
            if batch_graph is None:  # End signal
                print("Processing complete. Exiting graph processor...")
                break

            if batch_graph.number_of_nodes() == 0:
                print("Empty Graph. Skipping...")
                continue

            async with lock:
                existing_node_embeddings = update_node_embeddings({}, final_graph, embd_object)
                final_graph = nx.compose(final_graph, batch_graph) # Combine the graphs

                node_embeddings = update_node_embeddings(
                    existing_node_embeddings, final_graph, embd_object, remove_embeddings_for_nodes_no_longer_in_graph=True, verbatim=False
                )

            res = graph_statistics_and_plots_for_large_graphs(
                final_graph, data_dir=kwargs.get('data_dir_output', './data/GR_output_KG/'),
                include_centrality=False, make_graph_plot=False,
                root='final_graph'
            )

            # Save the merged graph as intermediate output
            async with lock:
                final_graph_path = f"{kwargs.get('data_dir_output', './data/GR_output_KG/')}/final_graph.graphml"
                await asyncio.to_thread(nx.write_graphml, final_graph, final_graph_path)

            print(f"Graph processed and added to the final graph. Current status: {res}")

        except asyncio.TimeoutError:
            print("Queue empty, terminating...")
            break
        except Exception as e:
            print(f"Error processing graph: {e}")
            traceback.print_exc()
        finally:
            graph_queue.task_done()

async def adjust_graph(graph, graph_embeddings, embd_object, **kwargs):
    """Adjust the graph and perform simplification."""
    G_new = graph.copy()
    node_embeddings = update_node_embeddings(graph_embeddings, G_new, embd_object)

    # Simplify the graph
    G_new, node_embeddings = simplify_graph(
        G_new, node_embeddings, embd_object,
        similarity_threshold=kwargs.get('similarity_threshold', 0.95),
        data_dir_output=kwargs.get('data_dir_output', './data/GR_output_KG/'),
        verbatim=False
    )

    # Keep only the largest connected component if specified
    if kwargs.get('return_only_giant_component', False):
        largest_component = max(nx.connected_components(G_new), key=len)
        G_new = G_new.subgraph(largest_component).copy()
        node_embeddings = update_node_embeddings(node_embeddings, G_new, embd_object, verbatim=False)

    # Perform Louvain clustering if needed
    if kwargs.get('do_Louvain_on_new_graph', True):
        G_new = graph_Louvain(G_new)

    return G_new, node_embeddings

async def main(all_docs, llm, embd):
    """Main coroutine to handle graph generation and processing."""
    batch_size = 10
    graph_queue = Queue()
    final_graph = nx.Graph()
    lock = Lock()
    exception_handler = []

    # Start the graph processor task
    processor_task = asyncio.create_task(graph_processor(graph_queue, final_graph, embd, lock))

    # Generate graphs in batches asynchronously
    tasks = []
    for i in range(0, len(all_docs), batch_size):
        batch_docs = all_docs[i:i + batch_size]
        tasks.append(graph_generator(batch_docs, llm, graph_queue))

    await asyncio.gather(*tasks, return_exceptions=True)

    # Signal the processor to terminate
    await graph_queue.put(None)
    await processor_task

    # Save the final graph
    final_graph_path = './data/final_graph.graphml'
    await asyncio.to_thread(nx.write_graphml, final_graph, final_graph_path)
    print(f"Final graph saved to {final_graph_path}")

    # Simplify the graph
    simple_graph, simple_node_embeddings = await adjust_graph(final_graph, exisiting_embeddings, embd, return_only_giant_component=False, do_Louvain_on_new_graph=True)

# Usage example
if __name__ == "__main__":
    pickle_file_path = './data/storage/full_all_documents.pkl'
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            all_docs = pickle.load(f)

    asyncio.run(main(all_docs, llm, embd))