
**Minimizing Retraining:** RAG reduces the need for periodic retraining. Instead, you can refresh the external database, keeping the AI system up-to-date without overhauling the model.

**Boosting Domain-Specific Responses:** RAG can draw from domain-specific databases, e.g., medical repositories, to provide detailed, accurate answers.

## **RAG vs. Fine-Tuning LLMs: A Practical Comparison**

Fine-tuning tailors an LLM for specific tasks by retraining it with niche data, like adapting a model for legal jargon or medical terminology. 

This method is effective but often locks the model to the knowledge it was trained on, which can become outdated

 contrast, RAG introduces a dynamic mechanism, continually integrating fresh, external information. For example, in a healthcare setting, RAG can pull the latest medical research or treatment guidelines

## **Retrieval Augmented Generation - Technical Requirements**

**Data Ingestion Pipeline Step:** In this phase, the system orchestrates the gathering of relevant data and converts it into embeddings. These processed embeddings are subsequently structured to provide the LLM model with the necessary context for generating responses.

**Retrieval Step:** At this step, the retrieval mechanism comes into play, pinpointing the segments of data that are most relevant from the available datasets.

**Generation Step:** Subsequently, the generation component, utilizing models akin to LLM, synthesizes a response that is both informed and contextually aligned with the data retrieved.

![image.png](https://s3.timeweb.cloud/4498e8c1-my-stuff/studyfair/report/diagram-1.png)

1. Data Collection: Obtaining raw data from various sources.
2. Pre-processing: Cleaning the data to remove any inconsistencies, irrelevant information, or errors. This step may involve normalization, tokenization, and other data transformation techniques.
3. Transformation using Embedding model: Converting data into a format that's amenable for use in the subsequent layers, converting text data into numerical vectors or embeddings. The main goal is to capture semantic relationships between words/phrases so that words with similar meanings are close in the embedding space.
4. Vector Store Insertion: Before insertion, vectors are often indexed to facilitate efficient retrieval. Finally, the indexed vectors are stored in the vector database.

### **Retrieval Step**

1. Query Processing: This is the initial stage where the system receives a query from the user.
- Input: Could be text, image, etc.
- Preprocessing: Similar to the data insertion pipeline, query data is preprocessed to match the format expected by the embedding model.
1. Query Embedding: The preprocessed query is converted into an embedding vector using the same model (or compatible one) that was used for generating embeddings during the insertion pipeline.
2. Similarity Search: The query embedding is then used to search the vector store for the nearest neighbors.
3. Candidate Generation: Based on the nearest neighbors, the system generates a set of candidate data points that could be relevant to the query.
4. Filtering & Ranking: Further filtering and ranking might be applied to the retrieved neighbors to select the best candidates.

In some systems, additional processing is applied to the candidates to generate the final output.

- LLM: A model such as [Llama2](https://ai.meta.com/llama/), [GPT](https://platform.openai.com/docs/models), [Mistral](https://mistral.ai/) could take the candidates and generate new data
- Aggregation: In cases like recommendations, the candidates are often aggregated to form a single coherent response.

The generated data or response might require post-processing before being presented to the user.

- Formatting: Ensuring the data is in a user-friendly format.
- Personalization: Tailoring the output to the user's preferences.

This layer manages how prompts are fed into the LLM to control its output or guide its generation process.

- Prompt Design: Designing prompts that guide the model to generate desired outputs. This can involve iterating and refining based on the model's responses.
- Sequential Interaction: Some tasks might require multiple prompts to be sent sequentially, with the model's output from one prompt being used as input for the next. This "chaining" can help in guiding the model towards a more refined or specific output.
- **Feedback Loop**: The chaining prompts layer might incorporate a feedback mechanism, where the model's output is analyzed, and subsequent prompts are adapted accordingly.

**Looking ahead, the evolution of RAG and LLM technologies is likely to focus on incremental improvements rather than revolutionary changes**
