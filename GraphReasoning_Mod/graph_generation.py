""" 
graph_generation.py

This module contains functions for generating graph concepts from text data using a language model.

Modified by Jonathan Kasprisin 

References: https://github.com/rahulnyk/knowledge_graph and https://github.com/lamm-mit/GraphReasoning
"""
from GraphReasoning_Mod.graph_tools import *
import uuid
import pandas as pd
import numpy as np
import networkx as nx
import os
from tqdm import tqdm
from typing import List
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json
from pathlib import Path
from pyvis.network import Network #for visualizing the graph
import re
import unicodedata
import time
import traceback
from IPython.display import display

def clean_text(text):
    """Remove problematic Unicode characters and replace them with safe alternatives."""
    if isinstance(text, str):
        # Normalize Unicode text (NFKD to decompose characters like é -> e)
        text = unicodedata.normalize("NFKD", text)

        # Define replacements for known problematic characters
        replacements = {
            "\u2212": "-",   # Unicode minus → Hyphen
            "\ufffd": "",    # Replacement character → Remove
            "\u2225": "||",  # Parallel symbol → Double pipes
            "\u03b1": "alpha",  # Greek alpha → "alpha"
            "\u03b8": "theta"   # Greek theta → "theta"
        }

        # Apply replacements
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        # Remove any remaining non-printable characters
        text = re.sub(r'[^\x20-\x7E\u0080-\uFFFF]', '', text).strip()

        return text
    return str(text).strip()  # Convert non-strings to string safely

def split_text_for_kg(documents: List[Document], chunk_size=2500,chunk_overlap=500,):
    """
    Cleans metadata and splits a list of documents into smaller chunks for knowledge graph generation.

    Args:
        documents (List[Document]): A list of Document objects.
        chunk_size (int): The maximum number of characters in each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Document]: A list of Document objects, each representing a text chunk.

    """
    # #filter complex meta data like lists from yt videos
    # simple_metadata_docs =  filter_complex_metadata(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        add_start_index=True, #track index in orginal document
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        # is_separator_regex=False,
    )

    #print("splitting documents...")
    all_splits = text_splitter.split_documents(documents)

    #prevent issues in metadata. Replace None values in metadata with a default value
    def clean_metadata(metadata):        
        for key, value in metadata.items():
            if value is None:
                metadata[key] = ""  # Replace None with an empty string or a default value
        return metadata

    for split in all_splits:
        split.metadata= clean_metadata(split.metadata)
    
    return all_splits
 

def extract (string, start='[', end=']'):
    """ Returns a substring from a string between two delimiters to help get json from llm output"""
    start_index = string.find(start)
    end_index = string.rfind(end)
    return string[start_index :end_index+1]

def documents2Dataframe(documents) -> pd.DataFrame:
    """
    Converts a list of document strings into a pandas DataFrame with unique chunk IDs.

    Args:
        documents (list): A list of text strings, each representing a document chunk.

    Returns:
        pd.DataFrame: A DataFrame where each row contains a text chunk and its unique ID.
    """
    rows = []
    for chunk in documents:
        row = {
            "text": chunk,
            "chunk_id": uuid.uuid4().hex,
            "source": chunk.metadata["source"],
            "metadata": chunk.metadata,
        }
        rows = rows + [row]
    df = pd.DataFrame(rows)
    return df

def concepts2Df(concepts_list) -> pd.DataFrame:
    """
    Converts a list of concepts into a DataFrame, ensuring consistent formatting and removing NaN values.

    Args:
        concepts_list (list): A list of dictionaries representing concepts.

    Returns:
        pd.DataFrame: A DataFrame with cleaned and lowercase concept entities.
    """
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()
    )
    return concepts_dataframe

#Modified by Jonathan Kasprisin to for use with the langchain_huggingface endpoint
#keep metadata in the table
def df2Graph(dataframe: pd.DataFrame, llm, repeat_refine=0, verbatim=False):
    """
    Converts a DataFrame of text chunks into a list of graph concepts using an LLM.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing text chunks.
        llm: The language model endpoint for generating concepts.
        repeat_refine (int): Number of times to refine the generated graph concepts.
        verbatim (bool): If True, print intermediate results for debugging.

    Returns:
        list: A flattened list of graph concepts.
    """
    if verbatim:
        tqdm.pandas(desc="Processing rows")
        results = dataframe.progress_apply(
            lambda row: generate_graph_triplets(row.text, llm, {
                "chunk_id": row.chunk_id,
                "source": row.source,
                "metadata": row.metadata,
            }, repeat_refine=repeat_refine, verbatim=verbatim), axis=1
        )
    else:
        results = dataframe.apply(
            lambda row: generate_graph_triplets(row.text, llm, {
                "chunk_id": row.chunk_id,
                "source": row.source,
                "metadata": row.metadata,
            }, repeat_refine=repeat_refine, verbatim=verbatim), axis=1
        )

    # Ensure we only keep valid results (remove None values)
    results = results[results.notnull()]  # Handles None values safely
    results = results.dropna()
    results = results.reset_index(drop=True)
    if results.empty:
        print("Warning: All triplet generation failed, returning an empty concept list.")
        return []  # Return an empty list instead of causing failure
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates contextual proximity between nodes in a DataFrame by analyzing co-occurrences.

    Args:
        df (pd.DataFrame): A DataFrame containing nodes and their edges.

    Returns:
        pd.DataFrame: A DataFrame with contextual proximity edges and their counts.
    """
    ## Melt the dataframe into a list of nodes
    df['node_1'] = df['node_1'].astype(str)
    df['node_2'] = df['node_2'].astype(str)
    df['edge'] = df['edge'].astype(str)
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2


def graph2Df(nodes_list) -> pd.DataFrame:
    """ Transform a list of graph nodes and their relationships into a clean and structured pandas DataFrame
        and remove NaN values.
    """
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: str(x).lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: str(x).lower())

    return graph_dataframe


#Modified by Jonathan Kasprisin to work in langchain and address learning path specific key concepts
def generate_graph_triplets(input: str, llm, metadata={}, repeat_refine=1, verbatim=False):
    """
    Generate ontology graphs using LangChain's HuggingFaceEndpoint.

    Args:
        input (str): The context text for generating the ontology.
        llm (HuggingFaceEndpoint): LangChain endpoint for generating text.
        metadata (dict): Additional metadata to include in the result.
        repeat_refine (int): Number of times to refine the ontology.
        verbatim (bool): Whether to print intermediate results.

    Returns:
        list: A list of ontology triplets with metadata.
    """
    try: 
        SYS_PROMPT_GRAPHMAKER = """
        You are a specialized system for extracting relationship networks from mathematical texts. 
        Your task is to identify key concepts and their specific relationships from the given context (delimited by ```). 

        Focus on practical connections, prerequisites, and dependencies between concepts.
        Extract relationships that are:
            Prerequisites (what must be understood first)
            Dependencies (how concepts rely on each other)
            Applications (how concepts are used in practice)
            Computational relationships (what is used to calculate what)

        Guidelines:
            Always include full context for mathematical objects (e.g., "equation x + y = 5" instead of just "equation (1)")
            Use complete descriptive phrases for variables (e.g., "variable x representing time" not just "x")
            Treat complete equations as single units
            Use specific, active verbs in relationships (e.g., "computes", "proves", "depends on")
            Include both theoretical connections and practical applications
            Capture learning progression and knowledge requirements
            Avoid using node 'linear algebra' and use the relevant concept from linear algebra

        Output Format:
            List of JSON objects with:
            'node_1': First concept/equation/process (must be self-contained and fully described)
            'node_2': Related concept/equation/process (must be self-contained and fully described)
            'edge': Specific relationship using active verbs

        Avoid:
            References without context (e.g., "equation (2)", "variable x", "this case")
            Generic relationships ("is related to", "connects with")
            Internal equation relationships
            Vague or overly broad connections
            Circular dependencies
            Using "linear algebra" as a node when a more specific concept (e.g., eigenvectors, matrix transformations, linear independence) is available. Only use "linear algebra" if no more precise term applies

        Examples:
            Context: The quadratic equation x² + 2x + 1 = 0 has one solution. Understanding the discriminant b² - 4ac helps determine the nature of solutions.
            [
                {"node_1": "quadratic equation x² + 2x + 1 = 0", "node_2": "single solution x = -1", "edge": "yields"},
                {"node_1": "discriminant formula b² - 4ac", "node_2": "nature of quadratic solutions", "edge": "determines"},
                {"node_1": "zero discriminant value", "node_2": "single repeated solution", "edge": "indicates presence of"}
            ]

            Context: In linear algebra, eigenvalues λ of a square matrix A are found by solving the characteristic equation det(A−λI)=0. The process involves setting up the equation, computing the determinant, and solving for λ.
            [
                {"node_1": "eigenvector equation Av = λv", "node_2": "characteristic equation det(A - λI) = 0", "edge": "leads to"},
                {"node_1": "characteristic equation det(A - λI) = 0", "node_2": "eigenvalues of matrix A", "edge": "is used to compute"},
                {"node_1": "computing the determinant", "node_2": "characteristic polynomial", "edge": "produces"},
            ]

            Context: The system of equations 3x + 2y = 8 and 6x + 4y = 15 is inconsistent. The second equation is a multiple of the first but has a different constant term.
            [
                {"node_1": "system of equations {3x + 2y = 8, 6x + 4y = 15}", "node_2": "inconsistent linear system", "edge": "represents"},
                {"node_1": "equation 6x + 4y = 15", "node_2": "scaled version of equation 3x + 2y = 8", "edge": "is a"},
                {"node_1": "different constant terms in proportional equations", "node_2": "system inconsistency", "edge": "causes"}
            ]

        Analyze the text carefully and produce around 10 triplets, making sure they reflect consistent relationships with fully contextualized terms.
        """

        USER_PROMPT = f"Context: ```{input}```\n\nOutput:"

        response = llm.invoke(SYS_PROMPT_GRAPHMAKER + "\n" + USER_PROMPT)

        CLEAN_SYS_PROMPT = """
        Do not repeat any instrtuctions only respond in this JSON format:
        [
            {
                "node_1": "A concept from extracted ontology",
                "node_2": "A related concept from extracted ontology",
                "edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"
            },
            {...}
        ]
        """
        CLEAN_USER_PROMPT = f"""
            Read this Context: ```{input}``` 

            Read this Ontology: ```{response}```

            Improve the ontology by ensuring consistent and concise labels, and correct any errors in formatting.
            """
        
        response = llm.invoke(CLEAN_SYS_PROMPT + "\n" + CLEAN_USER_PROMPT)
    

        # Refinement loop
        if repeat_refine > 0:
            range_iter = range(repeat_refine)
            if verbatim:
                range_iter = tqdm(range_iter, desc="Refining ontology")
            
            for _ in range_iter:
                REFINE_USER_PROMPT = f"""
                Insert up to 10 new triplets into the original ontology.
                Read this Context: ```{input}``` 

                Read this Ontology: ```{response}```

                Insert additional triplets to the original list, in the same JSON format. Repeat original AND new triplets while ensuring consistent and concise labels.
                """
            
                response = llm.invoke(CLEAN_SYS_PROMPT + "\n" + REFINE_USER_PROMPT)
            

        # Post-process the response
        try:
            response = extract(response)
            # Clean response text before JSON parsing
            cleaned_response = clean_text(response)

            # Load JSON after cleaning
            result = json.loads(cleaned_response.encode('utf-8').decode('utf-8'))

            # Apply clean_text to each item in the JSON list, but only for string values
            for item in result:
                for key, value in item.items():
                    if isinstance(value, str):
                        item[key] = clean_text(value)

            result = [dict(item, **metadata) for item in result]
        except json.JSONDecodeError:
            if verbatim: 
                print(f"generate_graph_triplets ->Error parsing JSON response: {response}")

            #append error response and item metadata  to saved file. if file doesnt exist, create it
            with open('error_log_generate_triplet.txt', 'a') as f:
                f.write(f"Error parsing METADATA:{metadata} RESPONSE:[{response}] \n")

            return None # Skip this iteration but do not stop the script
        
        if verbatim:
                        print ("---------------JSON good, moving to the next row-----------")
        return result
    
    except UnicodeDecodeError as e:
        print(f"generate_graph_triplets -> Encoding error: {e}")
        traceback.print_exc()
        
        with open('error_log_generate_triplet.txt', 'a', encoding='utf-8') as f:
            f.write(f"Encoding error with input: {input}\nException: {e}\n")
        
        return None  # Continue execution without resetting graph

#Helper function added to extract json from llm output
def add_source_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'source' relationship to a DataFrame of graph nodes.

    Args:
        df (pd.DataFrame): A DataFrame containing nodes and their relationships.

    Returns:
        pd.DataFrame: A DataFrame with the 'source' relationship added.
    """ 

    # Extract unique nodes and their corresponding sources from both node_1 and node_2
    unique_nodes_sources = pd.concat([
        df[['node_1', 'source', 'chunk_id', 'metadata']].rename(columns={'node_1': 'node'}),
        df[['node_2', 'source', 'chunk_id', 'metadata']].rename(columns={'node_2': 'node'})
    ]).drop_duplicates()

    # Create new rows using dictionary comprehension for better readability with source as node_1 and unique nodes as node_2
    new_rows_df = unique_nodes_sources.assign(
        node_1=unique_nodes_sources['source'],
        node_2=unique_nodes_sources['node'],
        edge='is source document of'
    )[['node_1', 'node_2', 'chunk_id', 'source', 'metadata', 'edge']]

    # Append new rows to the original DataFrame
    df = pd.concat([df, new_rows_df], ignore_index=True)
    
    return df


def make_graph_from_doc_batch(docs, llm, include_contextual_proximity=False, graph_root='make_graph',
                        chunk_size=2500,chunk_overlap=0,
                        repeat_refine=0,verbatim=False,
                        data_dir='.data/GR_output_KG/',
                        add_source_nodes = True
                         ):    
    """
    Creates an undirected knowledge graph from a batch of documents using a language model and optional contextual proximity.

    This function processes a collection of documents to generate a knowledge graph by extracting concepts and relationships. 
    The resulting graph is saved in various formats and can include contextual proximity information if specified.

    Parameters:
        docs (list or iterable): A batch of documents to process.
        llm (object): A language model used for extracting concepts and relationships.
        include_contextual_proximity (bool, optional): If True, adds edges based on contextual proximity between concepts. Defaults to False.
        graph_root (str, optional): The base name for the graph files. Defaults to 'graph_root'.
        chunk_size (int, optional): The size of text chunks for processing. Defaults to 2500.
        chunk_overlap (int, optional): Overlap between consecutive text chunks. Defaults to 0.
        repeat_refine (int, optional): Number of iterations for refining the graph. Defaults to 0.
        verbatim (bool, optional): If True, enables verbose logging and output. Defaults to False.
        data_dir (str, optional): Directory where output files will be stored. Defaults to '.data/GR_output_KG/'.
        simple (bool, optional): If True, skips additional processing and file outputs for nodes and edges. Defaults to True.

    Returns:
        tuple: 
            - graph_GraphML (str): The path to the saved GraphML file.
            - G (networkx.Graph): The constructed NetworkX graph object.

    Notes:
        - The function splits the input documents into chunks and extracts concepts and relationships using the language model.
        - Outputs include a GraphML file, optionally CSV and JSON files for nodes and edges, and a DataFrame of graph statistics.
        - The function supports community detection and assigns colors and groups to nodes for visualization.
    """
    ## data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)     
     
    outputdirectory = Path(f"./{data_dir}/") #where graphs are stored from graph2df function
    #create csv file if it doesnt exist f'{data_dir}/{graph_root}_pre_simplify.csv'
    if not os.path.exists(f'{data_dir}/{graph_root}_pre_simplify.csv'):
        with open(f'{data_dir}/{graph_root}_pre_simplify.csv', 'w') as f:
            pass
    

    chunks = split_text_for_kg(docs, chunk_size=chunk_size,chunk_overlap=chunk_overlap)

    # if verbatim:
    #     #display(Markdown (chunks[0])) #works only for text
    #     display(chunks[0])
    
    df = documents2Dataframe(chunks)

    try:
        concepts_list = df2Graph(df, llm, repeat_refine=repeat_refine, verbatim=verbatim)
    except Exception as e:
        print(f"concept list Error occurred: {e}")
        traceback.print_exc()
        return None, nx.Graph()

    # **Handle the case where no concepts were generated**
    if not concepts_list or len(concepts_list) == 0:
        print(f"Warning: No concepts extracted for {graph_root}. Returning an empty graph.")
        return None, nx.Graph() 

    dfg1 = graph2Df(concepts_list)

    #convert metadata to string
    dfg1['metadata'] = dfg1['metadata'].apply(lambda x: str(x))

    if add_source_nodes:
        dfg1 = add_source_relationship(dfg1)

    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)

    dfg1.to_csv(outputdirectory/f"{graph_root}_graph.csv", sep="|", index=False, encoding="utf-8")
    df.to_csv(outputdirectory/f"{graph_root}_chunks.csv", sep="|", index=False, encoding="utf-8")

    
    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1['count'] = 4 
      
    if verbatim:
        print("Shape of graph DataFrame: ", dfg1.shape)
        dfg1.head()
    
    if include_contextual_proximity:
        dfg2 = contextual_proximity(dfg1)
        dfg = pd.concat([dfg1, dfg2], axis=0)
        
    else:
        dfg=dfg1


    #convert metadata to string
    dfg['metadata'] = dfg['metadata'].apply(lambda x: str(x))
        
    # Group by node_1 and node_2 and aggregate chunk_id, edge, count, and metadata
    dfg = (
        dfg.groupby(["node_1", "node_2"])
        .agg({
            "chunk_id": ",".join,
            "edge": 'first', # ','.join, #changed from join because it didnt seem to provide additional information
            'count': 'sum',
            'metadata': 'first'  # keep first instance of the metadata
        })
        .reset_index()
    )

    #apply clean_text to node_1, node_2, edge and metadata
    dfg['node_1'] = dfg['node_1'].apply(clean_text)
    dfg['node_2'] = dfg['node_2'].apply(clean_text)
    dfg['edge'] = dfg['edge'].apply(clean_text)
    dfg['metadata'] = dfg['metadata'].apply(clean_text)
        
    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
    if verbatim:
        print ("Nodes shape: ", nodes.shape)
    
    

    G = nx.Graph()
    node_list=[]
    node_1_list=[]
    node_2_list=[]
    title_list=[]
    weight_list=[]
    chunk_id_list=[]
    metadata_list=[]
    
    ## Add nodes to the graph
    for node in nodes:
        clean_node = clean_text(node)
        G.add_node(
            str(clean_node)
        )
        node_list.append (clean_node)
    
    ## Add edges to the graph
    for _, row in dfg.iterrows():
        
        G.add_edge(
            row["node_1"],
            row["node_2"],
            title=row["edge"],
            metadata=row["metadata"],
            weight=row['count']/4
        )
        
        node_1_list.append (row["node_1"])
        node_2_list.append (row["node_2"])
        title_list.append (row["edge"])
        weight_list.append (row['count']/4)
        chunk_id_list.append (row['chunk_id'] )
        metadata_list.append (row['metadata'])


    try:
        df_graph = pd.DataFrame({"node_1": node_1_list, "node_2": node_2_list,"edge_list": title_list, "weight_list": weight_list, "metadata": metadata_list } )    
        with open(f'{data_dir}/{graph_root}_pre_simplify.csv', 'a', encoding='utf-8') as f:
            df_graph.to_csv(f, header=False, index=False, encoding='utf-8')
            
    except:
        print("make_graph_from_doc_batch Error in creating edge dataframe ans saving to csv")
            

    # Perform community detection
    try:
        communities_generator = nx.community.girvan_newman(G)
        next_level_communities = next(communities_generator)
        communities = sorted(map(sorted, next_level_communities))

        if verbatim:
            print("Number of Communities = ", len(communities))

        colors = colors2Community(communities)

        for index, row in colors.iterrows():
            G.nodes[row['node']]['group'] = row['group']
            G.nodes[row['node']]['color'] = row['color']
            G.nodes[row['node']]['size'] = G.degree[row['node']]

    except ValueError:
        print(f"Warning: Unable to detect communities for graph {graph_root}, skipping this step.")
        
    graph_GraphML=  f'{data_dir}/{graph_root}_graphML.graphml'  #  f'{data_dir}/resulting_graph.graphml',

    try:
        nx.write_graphml(G, graph_GraphML, encoding="utf-8")
    except:
        print("make_graph Error in writing graph to GraphML")
    
    #     res_stat=graph_statistics_and_plots_for_large_graphs(G, data_dir=data_dir,include_centrality=False,
    #                                                     make_graph_plot=False,)
    #     if verbatim: print ("Graph statistics: ", res_stat)
        
    return graph_GraphML, G

#Modified from add_new_subgraph_from_text by Jonathan Kasprisin to work with langchain_huggingface endpoint  
def add_new_subgraph_from_docs(
    input_docs, llm, embd, 
    data_dir_output='./data_temp/', verbatim=False, size_threshold=10,
    chunk_size=2500, do_Louvain_on_new_graph=True,
    include_contextual_proximity=False, repeat_refine=0,
    similarity_threshold=0.95, do_simplify_graph=True,
    return_only_giant_component=False, add_source_nodes = False,
    save_common_graph=False,G_exisiting=None, 
    graph_GraphML_exisiting=None, existing_node_embeddings=None
    ):
    """
    Add a new subgraph to an existing graph or create an initial graph from input documents using LangChain.
    
    This function processes a batch of documents, generates a new graph, combines it with an existing graph (if provided), 
    updates node embeddings, and optionally applies graph processing techniques like simplification, Louvain clustering, 
    and retaining only the largest connected component. The resulting graph is saved and analyzed for statistics.

    Parameters:
        input_docs (list): List of LangChain Document objects to generate the new graph from.
        llm (object): The large language model (LLM) used for graph generation and processing.
        embd (object): Embedding model used for node embedding updates and similarity checks.
        data_dir_output (str): Directory path for saving temporary and final graph files. Default is './data_temp/'.
        verbatim (bool): Whether to print detailed progress and debug information. Default is True.
        size_threshold (int): Minimum size of graph fragments to retain. Fragments smaller than this are removed. Default is 10.
        chunk_size (int): Size of document chunks for graph generation. Default is 2500.
        do_Louvain_on_new_graph (bool): Whether to apply Louvain clustering to the new graph. Default is True.
        include_contextual_proximity (bool): Include contextual proximity relationships in the graph. Default is False.
        repeat_refine (int): Number of refinement iterations for graph generation. Default is 0.
        similarity_threshold (float): Similarity threshold for simplifying the graph. Default is 0.95.
        do_simplify_graph (bool): Whether to simplify the graph by merging similar nodes. Default is True.
        return_only_giant_component (bool): If True, retain only the largest connected component of the graph. Default is False.
        save_common_graph (bool): If True, save a subgraph of common nodes between existing and new graphs. Default is False.
        G_exisiting (networkx.Graph): An existing graph to augment with the new graph. Default is None.
        graph_GraphML_exisiting (str): File path to an existing graph in GraphML format to load. Default is None.
        existing_node_embeddings (dict): Node embeddings of the existing graph for updating. Default is None.

    Returns:
        tuple:
            - G_new (networkx.Graph): The augmented or newly created graph.
            - node_embeddings (dict): Updated node embeddings for the graph.
            - res (dict): Statistical analysis and plots of the final graph.
    """ 
    try:
        G_new = None
        res = None

        # Ensure mutually exclusive options for G_to_add and graph_GraphML_to_add
        assert not (G_exisiting and graph_GraphML_exisiting), "Provide either G_exisiting or graph_GraphML_exisiting, not both."

        if verbatim:
            print("Starting process to create or load a new graph...")

        # Load the existing graph if provided, otherwise create a new one
        if G_exisiting is not None:
            if verbatim:
                print("Loading provided existing graph...")
            G = G_exisiting
        elif graph_GraphML_exisiting is not None:
            if verbatim:
                print(f"Loading existing graph from GraphML: {graph_GraphML_exisiting}")
            G = nx.read_graphml(graph_GraphML_exisiting)
        else:
            if verbatim:
                print("No existing graph provided. Creating an initial graph...")
            G = nx.Graph()
            existing_node_embeddings = {}

        # Generate new graph from input documents
        if verbatim:
            print("Generating new graph from input documents...")

        graph_GraphML_to_add, G_to_add = make_graph_from_doc_batch(
            input_docs, llm,
            include_contextual_proximity=include_contextual_proximity,
            graph_root='graph_new',
            chunk_size=chunk_size,
            repeat_refine=repeat_refine,
            verbatim=verbatim,
            data_dir=data_dir_output,
            add_source_nodes = add_source_nodes
        )
        if verbatim:
            print(f"New graph generated: {graph_GraphML_to_add}")

        # Combine the graphs
        if verbatim:
            print("Combining the existing graph with the new graph...")
        G_new = nx.compose(G, G_to_add)

        if save_common_graph:
            if verbatim:
                print("Identifying and saving common nodes...")
            common_nodes = set(G.nodes()).intersection(set(G_to_add.nodes()))
            subgraph = G_new.subgraph(common_nodes)
            common_graph_path = f'{data_dir_output}/common_nodes.graphml'
            nx.write_graphml(subgraph, common_graph_path, encoding="utf-8")
            if verbatim:
                print(f"Common nodes graph saved to {common_graph_path}")

        # Update node embeddings
        if verbatim:
            print("Updating node embeddings...")
        node_embeddings = update_node_embeddings(
            existing_node_embeddings, G_new, embd, remove_embeddings_for_nodes_no_longer_in_graph=True, verbatim=False
        )
        if verbatim:
            print("Node embeddings updated.")

        # Simplify graph
        if do_simplify_graph:
            if verbatim:
                print("Simplifying the graph...")
            G_new, node_embeddings = simplify_graph(
                G_new, node_embeddings, embd,
                similarity_threshold=similarity_threshold,
                data_dir_output=data_dir_output,
                verbatim=verbatim
            )

        # # Remove small fragments if size_threshold > 0
        # if size_threshold > 0:
        #     if verbatim:
        #         print("Removing small fragments...")
        #     G_new = remove_small_fragments(G_new, size_threshold=size_threshold)
        #     node_embeddings = update_node_embeddings(node_embeddings, G_new, embd, verbatim=False)

        # Keep only the largest connected component if specified
        if return_only_giant_component:
            if verbatim:
                print("Extracting the largest connected component...")
            largest_component = max(nx.connected_components(G_new), key=len)
            G_new = G_new.subgraph(largest_component).copy()
            node_embeddings = update_node_embeddings(node_embeddings, G_new, embd, verbatim=False)

        # Perform Louvain clustering if needed
        if do_Louvain_on_new_graph:
            if verbatim:
                print("Performing Louvain clustering...")
            G_new = graph_Louvain(G_new)

        # Save the final graph
        final_graph_path = f'{data_dir_output}/final_augmented_graph.graphml'
        nx.write_graphml(G_new, final_graph_path, encoding="utf-8")
        if verbatim:
            print(f"Final graph saved to {final_graph_path}")

        # Generate statistics and plots
        if verbatim:
            print("Generating graph statistics...")
        res = graph_statistics_and_plots_for_large_graphs(
            G_new, data_dir=data_dir_output,
            include_centrality=False, make_graph_plot=False,
            root='final_graph'
        )

        if verbatim:
            print("Graph processing complete.")

        return  G_new, node_embeddings, res

    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        return G_exisiting, existing_node_embeddings, None
    

def standardize_document_metadata(pdf_docs, yt_docs, blog_docs):
    """
    Standardizes the metadata for PDF, YouTube, and blog documents, specific to the linear algebra `data.ipynb` ingestion

    This function updates the metadata for each document in the provided lists of PDF, YouTube, and blog documents.
    It ensures that each document has the metadata fields 'source', 'title', 'author', 'source_type', and drops any other metadata fields.

    Args:
        pdf_docs (list): A list of Document objects representing PDF documents.
        yt_docs (list): A list of Document objects representing YouTube documents.
        blog_docs (list): A list of Document objects representing blog documents.

    Returns:
        tuple: A tuple containing the updated lists of PDF, YouTube, and blog documents.
    """

    # Reference information
    reference_info = {
        "Gilbert_Strang_Linear_Algebra_and_Its_Applicatio_230928_225121.pdf": {
            "title": "Linear Algebra and its Applications (4th ed.)",
            "author": "Gilbert Strang"
        },
        "Introduction to Applied Linear Algebra VMLS.pdf": {
            "title": "Introduction to Applied Linear Algebra",
            "author": "Venderberghe, Lieven; and Boyd, Stephen"
        },
        "Jim Hefferon linalgebra.pdf": {
            "title": "Linear Algebra(4th ed.)",
            "author": "Jim Hefferon"
        },
        "Steven Leon Linear-Algebra-with-Applications.pdf": {
            "title": "Linear Algebra with Applications (8th ed.)",
            "author": "Steven Leon"
        }
    }

    # Clean PDF Source Data
    for doc in pdf_docs:
        # Extract the file name from the source path
        file_name = os.path.basename(doc.metadata['source'])
        new_metadata = {
            "source": f"{file_name} - page: {doc.metadata['page']}",
            "source_type": 'Textbook_PDF',
            "title": reference_info.get(file_name, {}).get("title", ""),
            "author": reference_info.get(file_name, {}).get("author", "")
        }
        doc.metadata = new_metadata

    for doc in yt_docs:
        new_metadata = {
            "source": doc.metadata.get('source', ''),
            "source_type": 'youtube',
            "title": doc.metadata.get('title', ''),
            "author": doc.metadata.get('author', '')
        }
        doc.metadata = new_metadata

    for doc in blog_docs:
        new_metadata = {
            "source": doc.metadata.get('source', ''),
            "source_type": 'blog',
            "title": doc.metadata.get('title', ''),
            "author": doc.metadata.get('author', '')
        }
        doc.metadata = new_metadata

    return pdf_docs, yt_docs, blog_docs
