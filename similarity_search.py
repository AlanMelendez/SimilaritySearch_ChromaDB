# chromadb lets us store and search documents.
# embedding_functions provides the model that converts text into embeddings.
import chromadb
from chromadb.utils import embedding_functions


# Use a Sentence Transformers model to convert text into numbers (embeddings).
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create a client for interacting with ChromaDB.
client = chromadb.Client()

# A collection is similar to a table: it stores related documents.
collection_name = "my_grocery_collection"


def main():
    """Create a collection, add grocery documents, and search the collection."""
    try:
        # Create a collection and configure it to use cosine distance.
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "A collection for storing grocery data"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": embedding_function,
            },
        )
        print(f"Collection created: {collection.name}")

        # These are the documents that will be stored and searched.
        texts = [
            "fresh red apples",
            "organic bananas",
            "ripe mangoes",
            "whole wheat bread",
            "farm-fresh eggs",
            "natural yogurt",
            "frozen vegetables",
            "grass-fed beef",
            "free-range chicken",
            "fresh salmon fillet",
            "aromatic coffee beans",
            "pure honey",
            "golden apple",
            "red fruit",
        ]

        # Create one unique ID for each document.
        ids = []
        for number in range(len(texts)):
            ids.append(f"food_{number + 1}")

        # Create one metadata dictionary for each document.
        # The metadata at position 0 belongs to the document at position 0.
        metadatas = []
        for text in texts:
            metadatas.append({"source": "grocery_store", "item": text})

        # Store the documents, metadata, and IDs in the collection.
        # ChromaDB creates the document embeddings automatically.
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        # Retrieve the stored documents so we can display the total count.
        all_items = collection.get()
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        perform_similarity_search(collection)
    except Exception as error:
        print(f"Error: {error}")


def perform_similarity_search(collection):
    """Search for documents related to the word 'apple'."""
    try:
        # This is the text that we want to search for.
        query_term = "apple"

        results = collection.query(
            query_texts=[query_term],
            n_results=3,
        )

        # ChromaDB returns a list for each query. We submitted one query,
        # so [0] selects the results belonging to that first query.
        query_ids = results["ids"][0]
        query_distances = results["distances"][0]
        query_documents = results["documents"][0]

        if len(query_ids) == 0:
            print(f'No documents found similar to "{query_term}"')
            return

        print(f'Top {len(query_ids)} similar documents to "{query_term}":')

        # The ID, distance, and document at the same index belong together.
        for index in range(len(query_ids)):
            doc_id = query_ids[index]
            distance = query_distances[index]
            text = query_documents[index]

            if not text:
                text = "Text not available"

            print(
                f' - ID: {doc_id}, Text: "{text}", '
                f"Distance: {distance:.4f}"
            )
    except Exception as error:
        print(f"Error in similarity search: {error}")


if __name__ == "__main__":
    main()
