"""Module containing the SlumberChunker."""

from bisect import bisect_left
from itertools import accumulate
from typing import Any, Callable, List, Literal, Optional, Union

from tqdm import tqdm

from chonkie.genie import BaseGenie, GeminiGenie
from chonkie.types import Chunk, RecursiveRules

from .recursive import RecursiveChunker

PROMPT_TEMPLATE = """<task> You are given a set of texts between the starting tag <passages> and ending tag </passages>. Each text is labeled as 'ID `N`' where 'N' is the passage number. Your task is to find the first passage where the content clearly seperates from the previous passages in topic and/or semantics. </task>

<rules>
Follow the following rules while finding the splitting passage:
- Always return the answer as a JSON parsable object with the 'split_index' key having a value of the first passage where the topic changes.
- Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable.
- If no clear `split_index` is found, return N + 1, where N is the index of the last passage. 
</rules>

<passages>
{passages}
</passages>
"""

class SlumberChunker(RecursiveChunker):
    """SlumberChunker is a chunker based on the LumberChunker — but slightly different."""

    def __init__(self,
                 genie: Optional[BaseGenie] = None, 
                 tokenizer_or_token_counter: Union[str, Callable, Any] = "gpt2",
                 chunk_size: int = 1024,
                 rules: RecursiveRules = RecursiveRules(),
                 candidate_size: int = 32,
                 min_characters_per_chunk: int = 12,
                 return_type: Literal["chunks", "texts"] = "chunks", 
                 verbose: bool = False):
        """Initialize the SlumberChunker.

        Args:
            genie (Optional[BaseGenie]): The genie to use.
            tokenizer_or_token_counter (Union[str, Callable, Any]): The tokenizer or token counter to use.
            chunk_size (int): The size of the chunks to create.
            rules (RecursiveRules): The rules to use to split the candidate chunks.
            candidate_size (int): The size of the candidate splits that the chunker will consider.
            min_characters_per_chunk (int): The minimum number of characters per chunk.
            return_type (Literal["chunks", "texts"]): The type of output to return.
            verbose (bool): Whether to print verbose output.

        """
        # Now, this would set the value of the self.chunk_size to be the candidate_size
        super().__init__(tokenizer_or_token_counter, candidate_size, rules, min_characters_per_chunk, return_type)

        # Lazily import the dependencies
        self._import_dependencies()

        # If the genie is not provided, use the default GeminiGenie
        if genie is None:
            genie = GeminiGenie()

        # Since we can't name it self.chunk_size, we'll name it self.input_size
        self.input_size = chunk_size
        self.genie = genie
        self.template = PROMPT_TEMPLATE
        self.verbose = verbose

    # TODO: Fix the type error later
    def chunk(self, text: str) -> List[Chunk]: # type: ignore
        """Chunk the text with the SlumberChunker."""
        splits = self._recursive_chunk(text, level=0, start_offset=0)

        # Add the IDS to the splits
        split_texts = [f"ID {i}: " + split.text for (i, split) in enumerate(splits)]

        cumulative_token_counts = list(accumulate([0] + [split.token_count for split in splits]))
        
        if self.verbose:
            progress_bar = tqdm(
                total=len(splits),
                desc="🦛",
                unit="split",
                bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} splits processed [{elapsed}<{remaining}, {rate_fmt}] 🌱",
                ascii=" o",
            ) 

        chunks = []
        current_pos = 0
        current_token_count = 0
        while(current_pos < len(splits)):
            # bisect_left can return 0? No because input_size > 0 and first value is 0
            group_end_index = min(bisect_left(cumulative_token_counts, current_token_count + self.input_size) - 1, len(splits))

            if group_end_index == current_pos:
                group_end_index += 1

            prompt = self.template.format(passages="\n".join(split_texts[current_pos:group_end_index]))
            response = int(self.genie.generate(prompt, Split)['split_index'])

            # Make sure that the response doesn't bug out and return a index smaller 
            # than the current position
            if current_pos >= response:
                response = current_pos + 1

            chunks.append(Chunk(
                text="".join([split.text for split in splits[current_pos: response]]),
                start_index=splits[current_pos].start_index,
                end_index=splits[response - 1].end_index,
                token_count = sum([split.token_count for split in splits[current_pos: response]])
            ))

            current_token_count = cumulative_token_counts[response]
            current_pos = response

            if self.verbose:
                progress_bar.update(current_pos - progress_bar.n)

        return chunks

    def _import_dependencies(self) -> None:
        """Import the dependencies for the SlumberChunker."""
        try: 
            global BaseModel, Split
            from pydantic import BaseModel

            class Split(BaseModel): # type: ignore
                split_index: int
    
        except ImportError:
            raise ImportError("The SlumberChunker requires the pydantic library to be installed. Please install it using `pip install chonkie[genie]`.")

    def __repr__(self) -> str:
        """Return a string representation of the SlumberChunker."""
        return (f"SlumberChunker(genie={self.genie}," +
                f"tokenizer_or_token_counter={self.tokenizer}, " +
                f"chunk_size={self.input_size}, " +
                f"candidate_size={self.chunk_size}, " + # Since we inherit from the RecursiveChunker
                f"min_characters_per_chunk={self.min_characters_per_chunk}, " +
                f"return_type={self.return_type})" # type: ignore
            )