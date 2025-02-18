# Dynamic Network of Knowledge (DNoK) LLM Integration Project

## Overview

DNoK\_GraphRAG is a project focused on integrating large language models (LLMs) with graph theory to compare knowledge graph generation and knowledge retrieval. This repository provides the tools, datasets, and scripts required to generate, analyze, and interact with knowledge graphs.

### Key Components

- **data**: Includes various datasets and outputs related to graph analysis.
- **GraphReasoning\_Mod**: Contains the core modules for graph analysis, generation, and utility functions. These modules are adapted from [https://github.com/lamm-mit/GraphReasoning](https://github.com/lamm-mit/GraphReasoning).
- **Notebooks**: Jupyter notebooks for gathering data, generating knowledge graphs, and testing LLM reasoning in different RAG/GraphRAG architectures.

### Features

- Custom graph generation and analysis scripts.
- Preprocessed datasets for immediate use.
- Integration with LangChain for LLM-based reasoning.

## Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
  - [data](#data)
  - [GraphReasoning\_Mod](#graphreasoning_mod)
  - [Notebooks](#notebooks)
- [Installation](#installation)
- [Usage](#usage)
- [Notes](#notes)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Structure

### data

- **data\_output\_KG\_test/**: Outputs from knowledge graph tests. Subdirectories exist for different generation methodologies.
- **exp\_outputs/**: Experimental outputs.
  - `testing_output.txt`: Example testing results.
- **GR\_output\_KG/**: Graph reasoning output data.
- **pdfs/**: Source PDFs for graph generation.
- **storage/**: Stores processed data, embeddings, and databases.
  - `chroma_db_stella_1.5B/`: ChromaDB for embeddings.
  - `full_all_documents.pkl`: Pickle file with all document embeddings.
- **temp/**: Temporary working files during graph generation.
  - `final_augmented_graph.graphml`: Final augmented graph.
  - `simple_graph_graphML_simplified.graphml`: Simplified graph output.

### GraphReasoning\_Mod

- `graph_generation.py`: Generates knowledge graphs from input datasets.
- `graph_analysis.py`: Analyzes and visualizes generated graphs.
- `graph_tools.py`: Utility functions for graph manipulation.
- `utils.py`: Miscellaneous helper functions.

### Notebooks

- `graph_generation_archive/`: Notebooks for different ways to generate knowledge graphs. Generation outputs are saved to examine dropped documents and errors from individual documents.
- `data.ipynb`: Previews and preprocesses datasets.
- `LLM_GraphReasoning.ipynb`: Demonstrates LLM-based reasoning with graphs.
- `LLM_reasoning.ipynb`: Explores reasoning techniques using LLMs.

## Installation

1. Clone the repository:

   ```bash
   git clone https://gitlab.example.com/username/DNoK_GraphRAG.git
   cd DNoK_GraphRAG
   ```

2. Create a conda environment:

   ```bash
   conda create --name dnoK_env python=3.12
   conda activate dnoK_env
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Examples for data scraping, graph generation, and retrieval can be found in the notebooks.

## Notes

1. Graphs are generated differently and have inconsistencies to include:
  - reasoning for edge weights
  - node and edge labeling format (e.g. graph with no relations, edge relation called 'title', or edge relation call 'relation')
2. Notebooks may have incorrect directories for saving and loading files due to repo restructure.
3. LLM usage is set to use a HuggingFace Inference Endpoint for Text Generation.  

### Methodology
- GraphReasoning Generation
    1. Content is put into standardized format of langchain Document
    2. A batch of 10 documents is processed into a subgraph
      ->make subgraph
          1. splits documents to standard chunk size
          2. Uses an llm to make a concept list of node-relationship -node from each chunk
          3. creates undirected subgraph (No metadata currently attached to nodes)
    3. Subgraph is added to the common graph while addressing common nodes
    4. Graph is simplified finding similar nodes and merging them together #TODO worth reviewing 
- Graph Context retrieval
    1. Given a query, uses llm to determine n most relevant and unique keywords
    2. Uses node embeddings to find the most similar nodes in the graph (best nodes)
    3. finds the shortest path between the two best nodes
    4. creates a subgraph for all nodes within 2 hops of the shortest path
    5. context = the shortest path and all nodes within 2 hops of the path


### Concerns
- graph_retrieval
    1. best node similarities are all very high (i.e. .98 and above) so its difficult to get quality best fit nodes.
- graph generation
    1. many keywords and nodes that are extraneous; common words ("a") or too short (linear algebra --> is defined as -->a) or too specific (e.g Ux +b +5 = 147)
    2. what metadata to include and should it be retrieved/ added to the context
    4. what relationships are priority
- Other
  1. What is the impact of undirected graph?


## License (Need to confirm if want to keep the MIT license and add the file)

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- **GraphReasoning**: Basis for GraphReasoning\_Mod with methodologies, tools, and functions. [https://github.com/lamm-mit/GraphReasoning](https://github.com/lamm-mit/GraphReasoning)
- **LangChain**: Framework for integrating LLMs.
- **ChromaDB**: Database for embeddings and document storage.
- **OTHER/ Future**: 
  - HybridRAG: https://arxiv.org/html/2408.04948v1 
  - "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" by Darren Edge et al: https://arxiv.org/abs/2404.16130 ; reference implimentation: https://github.com/stephenc222/example-graphrag 
  - Potential to try: Mircrosoft GraphRAG: https://github.com/microsoft/graphrag and reference article: https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/

---

