<div align="center">

## 🦛 Chonkie Docs 📚

</div>

> "Ugh, writing docs is such a pain — I'm going to make chonkie so simple that people will just get it!"
> — @chonknick, probably

Unfortunately, we do need docs for Chonkie (we tried!). While official docs are available at [docs.chonkie.ai](https://docs.chonkie.ai), these docs are meant as an additional resource to help you get the most out of Chonkie. Since these docs live inside the repo, they are a bit more flexible and can be updated more frequently, and are also a bit more detailed. Furthermore, they are easy to edit with AI, so you can ask the AI to update them with examples, recipes, and more! (Haha, less work for the maintainers! 🤖)

> [!NOTE]
> Since these docs are a single markdown file, they make it ultra-simple to add into your LLM of choice to answer questions about Chonkie! Cool, huh? Yeah, Chonkie is super cool. 🦛✨

## Table of Contents

- [🦛 Chonkie Docs 📚](#-chonkie-docs-)
- [Table of Contents](#table-of-contents)
- [📦 Installation](#-installation)
  - [Optional Dependencies](#optional-dependencies)
- [Usage](#usage)
- [CHONKosophy](#chonkosophy)
- [Chunkers](#chunkers)
  - [`TokenChunker`](#tokenchunker)
  - [`SentenceChunker`](#sentencechunker)
  - [`RecursiveChunker`](#recursivechunker)
  - [`SemanticChunker`](#semanticchunker)
  - [`SDPMChunker`](#sdpmchunker)
- [Tokenizers](#tokenizers)
- [Embeddings](#embeddings)
  - [How to support a new embedding model or provider?](#how-to-support-a-new-embedding-model-or-provider)
- [Package Versioning](#package-versioning)

## 📦 Installation

Chonkie is available for direct installation from PyPI, via the following command:

```bash
pip install chonkie
```

We believe in the rule of **minimum default dependencies** and **Make-Your-Own-Package (MYOP)** principles, so Chonkie has a bunch of optional dependencies that you can configure to get the most out of your Chonkie experience. Though, we do realize that it might be a pain to configure, so you can just install it all with the following command:

```bash
pip install "chonkie[all]"
```

We detail the optional dependencies below.

### Optional Dependencies

You can install optional features using the `pip install "chonkie[feature]"` syntax. Here's a breakdown of the available features:

| Feature    | Description                                                                                             |
| :--------- | :------------------------------------------------------------------------------------------------------ |
| `hub`      | Interact with the Hugging Face Hub for models and configurations. Required to access `from_recipe` options in Chunkers. |
| `viz`      | Enables the `Visualizer` which allows for cool visuals on the terminal and HTML output.                 |
| `code`     | Required for `CodeChunker`. Installs `tree-sitter` and `magika`.                                        |
|  `model2vec` | Required to leverage `Model2VecEmbeddings` with the semantic and late chunkers.                       |
| `st`       | Use `sentence-transformers` for generating embeddings, enabling semantic chunking strategies.           |
| `openai`   | Integrate with OpenAI's API for `tiktoken` token counting and OpenAI embeddings.                        |
| `voyageai` | Use Voyage AI's embedding models.                                                                       |
| `cohere`   | Integrate with Cohere's embedding models.                                                               |
| `jina`     | Use Jina AI's embedding models.                                                                         |
| `semantic` | Enable semantic chunking capabilities, potentially leveraging `model2vec`.                              |
| `neural`   | Utilize local Hugging Face `transformers` models (with `torch`) for advanced NLP tasks.                 |
| `genie`    | Integrate with Google's Generative AI (Gemini) models for advanced functionalities.                     |
| `all`      | Install all optional dependencies for the complete Chonkie experience. Not recommended for prod.        |


> [!NOTE]
> You can install multiple features at once by passing a list of features to the `pip install` command. For example, `pip install "chonkie[hub,viz]"` will install the `hub` and `viz` features.

## Usage

Chonkie is designed to be ultra-simple to use. There are usually always 3 steps: Install, Import, and CHONK! We'll go over a simple example below.


First, let's install Chonkie. We only need the base package since we'll be using the `RecursiveChunker` for this example

```bash
pip install chonkie
```

Next, we'll import Chonkie and create a `Chonkie` object.
```python
from chonkie import RecursiveChunker

chunker = RecursiveChunker()
```

Now, we'll use the `chunk` method to chunk some text.

```python
text = "Hello, world!"
chunks = chunker(text)
```

And that's it! We've just chonked some text. The `chunks` object is a list of `Chunk` objects. We can print them out to see what we've got.

```python
# Print out the chunks
for chunk in chunks:
    print(chunk.text)
    print(chunk.token_count)
    print(chunk.start_index)
    print(chunk.end_index)
```

Refer to the types reference below for more information on the `Chunk` object.

## CHONKosophy

Chonkie truly believes that chunking should be simple to understand, easy to use and performant where it matters. It is fundamental to Chonkie's design principles. We truly believe that chunking should never be brought into the foreground of your codebase, and should be a primitive that you don't even think about. Just like how we don't think about the `for` loop or the `if` statement at the assembly level (sorry assembly devs 🤖).


## Chunkers 

Chunkers are the core of Chonkie. They are responsible for chunking the text into smaller, more manageable pieces. There are many different types of chunkers, each with their own unique properties and use cases. We'll go over the different types of chunkers below.

### `TokenChunker`

The `TokenChunker` is the most basic type of chunker. It simply splits the text into chunks of a given token length. It comes with the default installation of Chonkie.

**Parameters:**

- `tokenizer (Union[str, Any])`: The tokenizer to use. Defaults to `gpt2` with `tokenizers.Tokenizer`. You can also pass `character` or `word` to use the character or word tokenizer respectively. More details mentioned in the [Tokenizers](#tokenizers) section.
- `chunk_size (int)`: The number of tokens to chunk the text into. Defaults to `512`.
- `overlap (int)`: The number of tokens to overlap between chunks. Defaults to `0`.
- `return_type (Literal["texts", "chunks"])`: Whether to return the chunks as a list of texts or a list of `Chunk` objects. Defaults to `"chunks"`.

**Methods:**

- `chunk(text: str) -> Union[List[Chunk], List[str]]`: Chunks a string into a list of `Chunk` objects or a list of strings.
- `chunk_batch(texts: List[str]) -> Union[List[List[Chunk]], List[List[str]]]`: Chunks a list of strings into a list of lists of `Chunk` objects or a list of lists of strings.
- `__call__(text: str) -> Union[List[Chunk], List[str], List[List[Chunk]], List[List[str]]]`: Chunks a string into a list of `Chunk` objects or a list of strings.

**Examples:**

Here are a couple of examples on how to use the `TokenChunker` in practice.

<details>
<summary><strong>1. Basic Usage of `TokenChunker`</strong></summary>

```python
from chonkie import TokenChunker

chunker = TokenChunker()
chunks = chunker("Hello, world!")

# Print out the chunks
for chunk in chunks:
    print(chunk.text)
    print(chunk.token_count)
    print(chunk.start_index)
    print(chunk.end_index)
```

</details>

<details>
<summary><strong>2. Using `TokenChunker` with a custom tokenizer</strong></summary>

```python
from chonkie import TokenChunker

chunker = TokenChunker(tokenizer="gpt2")
chunks = chunker("Hello, world!")
```

</details>

<details>
<summary><strong>3. Chunking a batch of text</strong></summary>

```python
from chonkie import TokenChunker

batch = [
    "Hello, world!",
    "This is a test",
    "Chunking is fun!"
]

chunker = TokenChunker()
chunks = chunker.chunk_batch(batch)
```

</details>

<details>
<summary><strong>4. Visualizing the chunks with `Visualizer`</strong></summary>

```python
from chonkie import TokenChunker, Visualizer

chunker = TokenChunker()
chunks = chunker("Hello, world!")

viz = Visualizer()
viz(chunks)
```

</details>

### `SentenceChunker`

The `SentenceChunker` is a chunker that splits the text into sentences and then groups the sentences together into chunks based on a given `chunk_size`, where `chunk_size` is the maximum tokens a chunk can have. Given that it groups naturally occurring sentence together, it's `token_count` value is not as consistent as the `TokenChunker`. However, it makes an excellent choice for chunking well formatted text, being both simple and fast.

**Parameters:**

- `tokenizer_or_token_counter (Union[str, Callable, Any])`: The tokenizer or token counter to use. Defaults to `gpt2` with `tokenizers.Tokenizer`. You can also pass `character` or `word` to use the character or word tokenizer respectively. Additionally, you can also pass a `Callable` that takes in a string and returns the number of tokens in the string. More details mentioned in the [Tokenizers](#tokenizers) section.
- `chunk_size (int)`: The maximum number of tokens a chunk can have. Defaults to `512`.
- `chunk_overlap (int)`: The number of tokens to overlap between chunks. Defaults to `0`.
- `min_sentences_per_chunk (int)`: Minimum number of sentences per chunk. Defaults to `1`.
- `min_characters_per_sentence (int)`: Minimum number of characters per sentence. Defaults to `12`.
- `approximate (bool)`: [DEPRECATED] Whether to use approximate token counting. Defaults to `False`.
- `delim (Union[str, List[str]])`: Delimiters to split sentences on. Defaults to `[". ", "! ", "? ", "\n"]`.
- `include_delim (Optional[Literal["prev", "next"]])`: Whether to include delimiters in the current chunk (`"prev"`), the next chunk (`"next"`), or not at all (`None`). Defaults to `"prev"`.
- `return_type (Literal["texts", "chunks"])`: Whether to return the chunks as a list of texts or a list of `Chunk` objects. Defaults to `"chunks"`.

**Methods:**

- `chunk(text: str) -> Union[List[Chunk], List[str]]`: Chunks a string into a list of `SentenceChunk` objects or a list of strings.
- `chunk_batch(texts: List[str]) -> Union[List[List[Chunk]], List[List[str]]]`: Chunks a list of strings into a list of lists of `SentenceChunk` objects or a list of lists of strings. (Inherited)
- `from_recipe(name: str, lang: str, **kwargs) -> SentenceChunker`: Creates a `SentenceChunker` instance using pre-defined recipes from the [Chonkie Recipe Store](https://huggingface.co/datasets/chonkie-ai/recipes). This allows easy configuration for specific languages or splitting behaviors.
- `__call__(text: str) -> Union[List[Chunk], List[str], List[List[Chunk]], List[List[str]]]`: Chunks a string or list of strings. Calls `chunk` or `chunk_batch` depending on input type. (Inherited)

**Examples:**

Here are a couple of examples on how to use the `SentenceChunker` in practice.

<details>
<summary><strong>1. Basic Usage of `SentenceChunker`</strong></summary>

```python
from chonkie import SentenceChunker

# Initialize with default settings (gpt2 tokenizer, chunk_size 512)
chunker = SentenceChunker()

text = "This is the first sentence. This is the second sentence, which is a bit longer. And finally, the third sentence!"
chunks = chunker(text)

# Print out the chunks
for chunk in chunks:
    print(f"Text: {chunk.text}")
    print(f"Token Count: {chunk.token_count}")
    print(f"Start Index: {chunk.start_index}")
    print(f"End Index: {chunk.end_index}")
    print(f"Number of Sentences: {len(chunk.sentences)}") # SentenceChunk specific attribute
    print("-" * 10)
```

</details>

<details>
<summary><strong>2. Using `SentenceChunker` with custom delimiters and smaller chunk size</strong></summary>

```python
from chonkie import SentenceChunker

# Use custom delimiters and a smaller chunk size
chunker = SentenceChunker(
    chunk_size=20,
    delim=["\n", ". "], # Split on newlines and periods followed by space
    include_delim="next" # Include delimiter at the start of the next chunk
)

text = "Sentence one.\nSentence two.\nSentence three is very short."
chunks = chunker(text)

for chunk in chunks:
    print(chunk.text)
    print(f"Tokens: {chunk.token_count}\n---")
```

</details>

<details>
<summary><strong>3. Using `SentenceChunker.from_recipe`</strong></summary>

```python
from chonkie import SentenceChunker

# Requires "chonkie[hub]" to be installed
# Uses default recipe for English ('en')
chunker = SentenceChunker.from_recipe(lang="en", chunk_size=64)

text = "This demonstrates using a recipe. Recipes define delimiters. They make setup easy."
chunks = chunker(text)

for chunk in chunks:
    print(chunk.text)
    print(f"Tokens: {chunk.token_count}\n---")
```

</details>

<details>
<summary><strong>4. Visualizing the chunks with `Visualizer`</strong></summary>

```python
# Requires "chonkie[viz]" to be installed
from chonkie import SentenceChunker, Visualizer

chunker = SentenceChunker(chunk_size=30)
text = "Chunk visualization is helpful. It shows how the text is split. Let's see how this looks."
chunks = chunker(text)

viz = Visualizer()
viz(chunks) # Prints colored output to terminal or creates HTML
```

</details>

### `RecursiveChunker`

The `RecursiveChunker` is a more complex type of chunker that uses a recursive approach to chunk the text. It is a good choice for chunking text that is not well-suited for the `TokenChunker`.

### `SemanticChunker`

The `SemanticChunker` splits text into semantically coherent chunks using sentence embeddings. It first splits the text into sentences, embeds them, and then groups sentences based on their semantic similarity. This approach aims to keep related sentences together within the same chunk, leading to more contextually meaningful chunks compared to fixed-size or simple delimiter-based methods. It's particularly useful for processing text where preserving the flow of ideas is important.

There are two main strategies for chunking:

1. **Window Strategy**: This strategy compares each sentence to the previous one (or within a small window) to determine if they are semantically similar. If they are, they are grouped together. Since it only compares a pre-defined window of sentences every time, it is easy to batch embed the (window, sentence) pairs and compare their similarity values.
2. **Cumulative Strategy**: This strategy compares each sentence to the mean embedding of the current group. If the sentence is more similar to the mean than the threshold, it is added to the group. Otherwise, a new group is started. This is much more computationally expensive than the window strategy, but can at times result in better chunks.

For both of the above strategies, in `auto` mode, we determine the `threshold` value based on a binary search over the range of values that keeps the median `chunk_size` below the `chunk_size` paramater and above the `min_chunk_size` parameter. While this may not always result in the ideal chunks, it does provide a good starting point. Hopefully, this will be improved in future versions of Chonkie.

**Parameters:**

- `embedding_model (Union[str, BaseEmbeddings])`: The embedding model to use for semantic chunking. Can be a string identifier (e.g., from Hugging Face Hub like `"minishlab/potion-base-8M"`) or an instantiated `BaseEmbeddings` object. Defaults to `"minishlab/potion-base-8M"`. Requires appropriate extras like `chonkie[semantic]` or specific model providers (`chonkie[st]`, `chonkie[openai]`, etc.).
- `mode (str)`: The strategy for comparing sentence similarity. `"window"` compares adjacent sentences (or within a small window), while `"cumulative"` compares a new sentence to the mean embedding of the current group. Defaults to `"window"`.
- `threshold (Union[str, float, int])`: The similarity threshold for splitting sentences. Can be `"auto"` (uses a binary search to find an optimal threshold based on `chunk_size`), a float between 0.0 and 1.0 (direct cosine similarity threshold), or an int between 1 and 100 (percentile threshold). Defaults to `"auto"`.
- `chunk_size (int)`: The target maximum number of tokens per chunk. Defaults to `512`.
- `similarity_window (int)`: When `mode="window"`, this defines the number of preceding sentences to consider when calculating the similarity of the current sentence. Defaults to `1`.
- `min_sentences (int)`: The minimum number of sentences allowed in a chunk. Defaults to `1`.
- `min_chunk_size (int)`: The minimum number of tokens allowed in a chunk. Also influences the minimum sentence length considered during splitting. Defaults to `2`.
- `min_characters_per_sentence (int)`: Minimum number of characters a sentence must have to be considered valid during the initial sentence splitting phase. Shorter segments might be merged. Defaults to `12`.
- `threshold_step (float)`: Step size used in the binary search when `threshold="auto"`. Defaults to `0.01`.
- `delim (Union[str, List[str]])`: Delimiters used to split the text into initial sentences. Defaults to `[". ", "! ", "? ", "\n"]`.
- `include_delim (Optional[Literal["prev", "next"]])`: Whether to include the delimiter with the preceding sentence (`"prev"`), the succeeding sentence (`"next"`), or not at all (`None`). Defaults to `"prev"`.
- `return_type (Literal["texts", "chunks"])`: Whether to return the chunks as a list of strings (`"texts"`) or a list of `SemanticChunk` objects (`"chunks"`). Defaults to `"chunks"`.

**Methods:**

- `chunk(text: str) -> Union[List[SemanticChunk], List[str]]`: Chunks a single string into a list of `SemanticChunk` objects or strings based on `return_type`.
- `chunk_batch(texts: List[str]) -> Union[List[List[SemanticChunk]], List[List[str]]]`: Chunks a list of strings. (Inherited)
- `from_recipe(name: str, lang: str, **kwargs) -> SemanticChunker`: Creates a `SemanticChunker` using pre-defined recipes (delimiters, etc.) from the [Chonkie Recipe Store](https://huggingface.co/datasets/chonkie-ai/recipes), simplifying setup for specific languages. Requires `chonkie[hub]`.
- `__call__(text: Union[str, List[str]]) -> Union[List[SemanticChunk], List[str], List[List[SemanticChunk]], List[List[str]]]`: Convenience method calling `chunk` or `chunk_batch` depending on input type. (Inherited)

**Examples:**

Here are a couple of examples on how to use the `SemanticChunker` in practice.

<details>
<summary><strong>1. Basic Usage of `SemanticChunker`</strong></summary>

```python
# Requires "chonkie[semantic]" or relevant embedding model extra (e.g., "chonkie[st]")
from chonkie import SemanticChunker

# Initialize with default settings (potion-base-8M model, auto threshold)
chunker = SemanticChunker()

text = "Semantic chunking groups related ideas. This sentence is related to the first. This one starts a new topic. Exploring different chunking strategies is key."
chunks = chunker(text)

# Print out the chunks (SemanticChunk objects)
for chunk in chunks:
    print(f"Text: {chunk.text}")
    print(f"Token Count: {chunk.token_count}")
    print(f"Start Index: {chunk.start_index}")
    print(f"End Index: {chunk.end_index}")
    print(f"Number of Sentences: {len(chunk.sentences)}") # SemanticChunk specific attribute
    print("-" * 10)
```

</details>

<details>
<summary><strong>2. Using `SemanticChunker` with a specific threshold and different model</strong></summary>

```python
# Requires "chonkie[semantic, st]" for sentence-transformers
from chonkie import SemanticChunker

# Use a different embedding model and a fixed percentile threshold
chunker = SemanticChunker(
    embedding_model="all-MiniLM-L6-v2", # From sentence-transformers
    threshold=90, # Use 90th percentile for similarity threshold
    chunk_size=128
)

text = "Using a percentile threshold can adapt to document density. 90 means splits occur at lower similarity points. This can result in more, smaller chunks potentially. Let's test this."
chunks = chunker(text)

for chunk in chunks:
    print(chunk.text)
    print(f"Tokens: {chunk.token_count}\n---")
```

</details>

<details>
<summary><strong>3. Using `SemanticChunker.from_recipe`</strong></summary>

```python
# Requires "chonkie[hub, semantic]" or relevant embedding model extra
from chonkie import SemanticChunker

# Uses default recipe for English ('en') delimiters
# Specify embedding model and other parameters as needed
chunker = SemanticChunker.from_recipe(
    lang="en",
    embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2", # Example
    chunk_size=64,
    threshold="auto"
)

text = "Recipes simplify delimiter setup. Semantic logic remains. This is English text."
chunks = chunker(text)

for chunk in chunks:
    print(chunk.text)
    print(f"Tokens: {chunk.token_count}\n---")
```

</details>

<details>
<summary><strong>4. Visualizing the chunks with `Visualizer`</strong></summary>

```python
# Requires "chonkie[viz, semantic]" or relevant embedding model extra
from chonkie import SemanticChunker, Visualizer

chunker = SemanticChunker(chunk_size=50)
text = "Visualization helps understand semantic breaks. See where the model decided to split the text based on meaning. This is useful for debugging."
chunks = chunker(text)

viz = Visualizer()
viz(chunks) # Prints colored output to terminal or creates HTML
```

</details>

### `SDPMChunker`

The `SDPMChunker` (Semantic Double-Pass Merging Chunker) builds upon the `SemanticChunker` by adding a second merging pass. After the initial semantic grouping of sentences, it attempts to merge nearby groups based on their semantic similarity, even if they are separated by a few other groups (controlled by the `skip_window` parameter). This can help capture broader semantic contexts that might be missed by only looking at immediately adjacent sentences or groups. It inherits most parameters and functionalities from `SemanticChunker`.

**Parameters:**

Inherits all parameters from `SemanticChunker` with the addition of:

- `skip_window (int)`: The number of groups to "skip" when checking for potential merges in the second pass. For example, with `skip_window=1`, the chunker compares group `i` with group `i+2`. Defaults to `1`.

**Methods:**

Inherits all methods from `SemanticChunker`, including:

- `chunk(text: str) -> Union[List[SemanticChunk], List[str]]`: Chunks a single string using the double-pass merging strategy.
- `chunk_batch(texts: List[str]) -> Union[List[List[SemanticChunk]], List[List[str]]]`: Chunks a list of strings. (Inherited)
- `from_recipe(name: str, lang: str, **kwargs) -> SDPMChunker`: Creates an `SDPMChunker` using pre-defined recipes. Requires `chonkie[hub]`.
- `__call__(text: Union[str, List[str]]) -> Union[List[SemanticChunk], List[str], List[List[SemanticChunk]], List[List[str]]]`: Convenience method. (Inherited)

**Examples:**

Here are a couple of examples on how to use the `SDPMChunker` in practice.

<details>
<summary><strong>1. Basic Usage of `SDPMChunker`</strong></summary>

```python
# Requires "chonkie[semantic]" or relevant embedding model extra (e.g., "chonkie[st]")
from chonkie import SDPMChunker

# Initialize with default settings (potion-base-8M model, auto threshold, skip_window=1)
chunker = SDPMChunker()

text = "This is the first topic. It discusses semantic chunking. This is a related sentence. Now we switch to a second topic. This topic is about embeddings. We go back to the first topic now. Double-pass merging helps here."
chunks = chunker(text)

# Print out the chunks (SemanticChunk objects)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(f"Text: {chunk.text}")
    print(f"Token Count: {chunk.token_count}")
    print(f"Start Index: {chunk.start_index}")
    print(f"End Index: {chunk.end_index}")
    print(f"Number of Sentences: {len(chunk.sentences)}")
```

</details>

<details>
<summary><strong>2. Using `SDPMChunker` with a larger `skip_window`</strong></summary>

```python
# Requires "chonkie[semantic, st]" for sentence-transformers
from chonkie import SDPMChunker

# Use a larger skip window and a specific model
chunker = SDPMChunker(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=128,
    skip_window=2 # Try merging groups i and i+3
)

text = "Topic A, sentence 1. Topic A, sentence 2. Topic B, sentence 1. Topic C, sentence 1. Topic A, sentence 3. Merging across B and C might occur."
chunks = chunker(text)

for chunk in chunks:
    print(chunk.text)
    print(f"Tokens: {chunk.token_count}\n---")
```

</details>


## Tokenizers

Fundamentally, chunking is a token-based operation. Chunking is done to load chunks into embedding models or LLMs, and limitations around size are often token-based. Chonkie supports a variety of tokenizers and tokenizer engines, through its `Tokenizer` class. 

The `Tokenizer` class is a wrapper that holds the `tokenizer` engine object and provides a unified interface to `encode`, `decode` and `count_tokens`.

**Available Tokenizers:**

- `character`: Character tokenizer that encodes characters.
- `word`: Word tokenizer that encodes words.
- `tokenizers`: Allows loading any tokenizer from the Hugging Face `tokenizers` library.
- `tiktoken`: Allows using the `tiktoken` tokenizer from OpenAI.
- `transformers`: Allows loading tokenizers from `AutoTokenizer` within the `transformers` library.

**Usage:**

You can initialize a `Tokenizer` object with a string that maps to the desired tokenizer.

```python
from chonkie import Tokenizer

tokenizer = Tokenizer("gpt2")
```

You can also pass a `tokenizer` engine object to the `Tokenizer` constructor.

```python
from tiktoken import get_encoding
from chonkie import Tokenizer

# Get the tiktoken encoding for gpt2
encoding = get_encoding("gpt2")

# Initialize the Tokenizer with the encoding
tokenizer = Tokenizer(tokenizer=encoding)
```

**Methods:**

- `encode(text: str) -> List[int]`: Encodes a string into a list of tokens.
- `encode_batch(texts: List[str]) -> List[List[int]]`: Encodes a list of strings into a list of lists of tokens.
- `decode(tokens: List[int]) -> str`: Decodes a list of tokens into a string.
- `decode_batch(tokens: List[List[int]]) -> List[str]`: Decodes a list of lists of tokens into a list of strings.
- `count_tokens(text: str) -> int`: Counts the number of tokens in a string.
- `count_tokens_batch(texts: List[str]) -> List[int]`: Counts the number of tokens in a list of strings.

**Example:**

```python
from chonkie import Tokenizer

tokenizer = Tokenizer("gpt2")

tokens = tokenizer.encode("Hello, world!")
print(tokens)

decoded = tokenizer.decode(tokens)
print(decoded)

token_count = tokenizer.count_tokens("Hello, world!")
print(token_count)
```

## Embeddings

Chonkie has quite a few usecases for embeddings —— `SemanticChunker` uses them to embed sentences, `LateChunker` uses them to get token embeddings, and the `EmbeddingsRefinery` uses them to get embeddings for downstream upsertion into vector databases. Chonkie tries to support a variety of different embedding models, and providers so that it can be used by as many people as possible.

**Available Embedding Models:** Chonkie supports the following embedding models (with their aliases):

- `Model2VecEmbeddings` (`model2vec`): Uses the `Model2Vec` model to embed text.
- `SentenceTransformerEmbeddings` (`sentence-transformers`): Uses a `SentenceTransformer` model to embed text.
- `OpenAIEmbeddings` (`openai`): Uses the OpenAI embedding API to embed text.
- `CohereEmbeddings` (`cohere`): Uses Cohere's embedding API to embed text.
- `JinaEmbeddings` (`jina`): Uses Jina's embedding API to embed text.
- `VoyageAIEmbeddings` (`voyageai`): Uses the Voyage AI embedding API to embed text.

Given that it has a bunch of different embedding models, it becomes challenging to keep track of which `Embeddings` class can load a given model. To make this easier, we built the `AutoEmbeddings` class. With `AutoEmbeddings`, you can pass a URI string of the model you want to load and it will return the appropriate `Embeddings` class. The URI usually takes the form of `alias://model_name` or `alias://provider/model_name`.

```python
from chonkie import AutoEmbeddings

# Since this model is registered with the Registry, we can use the string directly
embeddings = AutoEmbeddings("minishlab/potion-base-8M")

# If it's not registered, we can use the full URI
embeddings = AutoEmbeddings.get_embedding("model2vec://minishlab/potion-base-32M")
```

If you're trying to load a model from a local path, it's recommended to use the `SentenceTransformerEmbeddings` class. With the `AutoEmbeddings` class, you can pass in the `model` object initialized with the `SentenceTransformer` class as well, and it will return chonkie's `SentenceTransformerEmbeddings` object. 

> [!NOTE]
> If `AutoEmbeddings` can't find a model, it will try to search the HuggingFace Hub for the model and load it with the `SentenceTransformerEmbeddings` class. If that also fails, it will raise a `ValueError`.

**Methods:**

All `Embeddings` classes have the following methods:

- `embed(text: str) -> List[float]`: Embeds a string into a list of floats.
- `embed_batch(texts: List[str]) -> List[List[float]]`: Embeds a list of strings into a list of lists of floats.
- `get_tokenizer_or_token_counter() -> Any`: Returns the tokenizer or token counter object.
- `__call__(text: str) -> List[float]`: Embeds a string into a list of floats.

**Example:**

```python
from chonkie import AutoEmbeddings

# Get the embeddings for a model
embeddings = AutoEmbeddings("minishlab/potion-base-8M")

# Embed a string
embedding = embeddings("Hello, world!")
```

### How to support a new embedding model or provider? 

If you're trying to load a model that is not already supported by Chonkie, don't worry! We've got you covered. Just follow the steps below:

1. Check if your provider supports the OpenAI API. If it does, you can use the `OpenAIEmbeddings` class with the `base_url` parameter to point to your provider's API. You're all set!
2. If your provider does not support the OpenAI API, and you're loading a model locally, you can use the `SentenceTransformerEmbeddings` class to load your model. You'll need to pass in the `model` object initialized with your model.
3. Lastly, you can create your own `Embeddings` class by inheriting from the `BaseEmbeddings` class and implementing the `embed`, `embed_batch`, and `get_tokenizer_or_token_counter` methods.

**Example:**

```python
from typing import List, Any
from chonkie import BaseEmbeddings

# Let's say we have a custom embedding model that we want to support
class MyEmbeddings(BaseEmbeddings):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed(self, text: str) -> List[float]:
        return self.model.embed(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_batch(texts)
    
    def get_tokenizer_or_token_counter(self) -> Any:
        return self.tokenizer

    @property
    def dimension(self) -> int:
        return self.model.dimension

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, tokenizer={self.tokenizer})"

```

Of course the above example is a bit contrived, but you get the idea. Once you're done, you can use the above `Embeddings` class with the `SemanticChunker` or `LateChunker` classes, and it will work as expected!



## Package Versioning

Chonkie doesn't fully comply with Semantic Versioning. Instead, it uses the following convention:

- `MAJOR`: Serious refactoring or complete rewrite of the package.
- `MINOR`: Breaking changes to the package.
- `PATCH`: Bug fixes, new features, performance improvements, etc.
