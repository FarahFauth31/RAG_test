import os
from llmware.library import Library
from llmware.retrieval import Query
from llmware.setup import Setup
from llmware.status import Status
from llmware.prompts import Prompt
from llmware.configs import LLMWareConfig, MilvusConfig
from importlib import util

#Check if all libraries we need are installed
if not util.find_spec("torch") or not util.find_spec("transformers"):
    print("\nto run this example, with the selected embedding model, please install transformers and torch, e.g., "
          "\n`pip install torch`"
          "\n`pip install transformers`")

if not (util.find_spec("chromadb") or util.find_spec("pymilvus") or util.find_spec("lancedb") or util.find_spec("faiss")):
    print("\nto run this example, you will need to pip install the vector db drivers. see comments above.")


def semantic_rag (library_name, embedding_model_name, llm_model_name):

    """Semantic similarity query using RAG"""

    #First we create a library which will contain all the dcuments we want to have access to (parsing, text chunking and embedding)
    print ("\nupdate: Step 1 - Creating library: {}".format(library_name))

    library = Library().create_new_library(library_name)

    #Dowloading sample files, in this case some veterinary reports
    print ("update: Step 2 - Downloading Sample Files")

    sample_files_path = SAMPLE_FILES_PATH
    contracts_path = os.path.join(sample_files_path, "Vet_reports")

    #Parsing and text indexing files
    print("update: Step 3 - Parsing and Text Indexing Files")

    library.add_files(input_folder_path=contracts_path, chunk_size=400, max_chunk_size=800, smart_chunking=2)
    #library = Library().load_library("example1_library")

    #Install the embeddings
    print("\nupdate: Step 4 - Generating Embeddings in {} db - with Model- {}".format(vector_db, embedding_model))

    library.install_new_embedding(embedding_model_name=embedding_model_name, vector_db=vector_db, batch_size=200)

    # RAG steps start here ...

    print("\nupdate: Loading model for LLM inference - ", llm_model_name)

    prompter = Prompt().load_model(llm_model_name, temperature=0.0, sample=False)

    query = "find animal and animal breed"

    #Run semantic similarity query against the library and get all of the top results
    results = Query(library).semantic_query(query, result_count=80, embedding_distance_threshold=1.0)

    #   if you want to look at 'results', uncomment the line below
    # for i, res in enumerate(results): print("\nupdate: ", i, res["file_source"], res["distance"], res["text"])

    for i, contract in enumerate(os.listdir(contracts_path)):

        qr = []

        if contract != ".DS_Store":

            print("\nContract Name: ", i, contract)

            #Look through the list of semantic query results, and pull the top results for each file
            for j, entries in enumerate(results):

                library_fn = entries["file_source"]
                if os.sep in library_fn:
                    # handles difference in windows file formats vs. mac / linux
                    library_fn = library_fn.split(os.sep)[-1]

                if library_fn == contract:
                    print("Top Retrieval: ", j, entries["distance"], entries["text"])
                    qr.append(entries)

            #Add the query results to the prompt
            source = prompter.add_source_query_results(query_results=qr)

            response = prompter.prompt_with_source(query, prompt_name="default_with_context")

            for resp in response:
                if "llm_response" in resp:
                    print("\nupdate: llm answer - ", resp["llm_response"])

            # start fresh for next document
            prompter.clear_source_materials()

    return 0


if __name__ == "__main__":

    LLMWareConfig().set_active_db("sqlite")

    embedding_model = "jina-small-en-v2"

    MilvusConfig().set_config("lite", True)

    #   select one of:  'milvus' | 'chromadb' | 'lancedb' | 'faiss'
    LLMWareConfig().set_vector_db("chromadb")

    vector_db = "chromadb"

    lib_name = "example_5_library"

    example_models = ["bling-phi-3-gguf", "llmware/bling-1b-0.1", "llmware/dragon-yi-6b-gguf"]

    llm_model_name = example_models[0]

    semantic_rag(lib_name, embedding_model, llm_model_name)
