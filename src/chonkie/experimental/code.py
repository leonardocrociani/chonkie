"""Experimental code chunker with advanced AST-based parsing.

This module provides an experimental CodeChunker class that uses tree-sitter
for advanced code analysis and language-specific chunking strategies.
"""
import importlib.util as importutil
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from chonkie.chunker.base import BaseChunker
from chonkie.types.base import Chunk
from chonkie.types.code import SplitRule

from .code_registry import CodeLanguageRegistry

if TYPE_CHECKING:
  from tree_sitter import Node

class CodeChunker(BaseChunker):
  """Experimental code chunker using tree-sitter for advanced AST-based parsing.
  
  This chunker provides intelligent code splitting by understanding the syntax tree
  structure of various programming languages. It uses language-specific rules to
  maintain semantic coherence while respecting chunk size constraints.
  
  Features:
    - AST-based parsing using tree-sitter
    - Automatic language detection using Magika
    - Language-specific merge and split rules
    - Intelligent grouping of related code elements
    - Support for multiple programming languages
    - Optional chunk size management with recursive splitting
  
  Args:
    language (str): The programming language to parse. Defaults to "auto" for 
      automatic detection. Supported languages: python, typescript, javascript, 
      rust, go, java, markdown, html, css, c, cpp, csharp.
    tokenizer_or_token_counter (str): The tokenizer to use for token counting.
      Defaults to "character".
    chunk_size (Optional[int]): Maximum chunk size threshold. When specified,
      large code constructs will be recursively split to respect this limit.
      Note that chunks may exceed this size to preserve semantic structure
      and code coherence. Defaults to None (no size limit).
    add_split_context (bool): Whether to add contextual information about
      the split location to chunks. Defaults to True.

  """

  def __init__(self, language: str = "auto", tokenizer_or_token_counter: str = "character", chunk_size: Optional[int] = None, add_split_context: bool = True) -> None:
    """Initialize the CodeChunker.

    Args:
      language (str): The language to chunk.
      chunk_size (Optional[int]): Maximum chunk size threshold. Chunks may exceed 
        this size to preserve semantic structure and code coherence. Defaults to None.
      tokenizer_or_token_counter (str): The tokenizer to use.
      add_split_context (bool): Whether to add the split context to the chunks.

    """
    super().__init__(tokenizer_or_token_counter=tokenizer_or_token_counter)

    # Import the dependencies
    self._import_dependencies()

    # Initialize the state
    self.chunk_size = chunk_size
    self.add_split_context = add_split_context
    self.language = language

    # Initialize parser and language config based on language parameter
    if language == "auto":
      # Set a warning to the user that the language is auto and this might
      # affect the performance of the chunker.
      warnings.warn("The language is set to `auto`. This would adversely affect the performance of the chunker. " +
                   "Consider setting the `language` parameter to a specific language to improve performance.")
      
      # Set the language to auto and initialize the Magika instance
      self.magika = Magika() # type: ignore
      self.parser = None
      self.language_config = None
    else:
      # Check if the language has been defined or not
      if language not in CodeLanguageRegistry:
        raise ValueError(f"Language {language} is not registered in the configs.")
      
      self.parser = get_parser(language) # type: ignore
      self.language_config = CodeLanguageRegistry[language]

  def _import_dependencies(self) -> None:
    """Import the dependencies from the node."""
    try:
        global Node, get_parser, Magika
        from tree_sitter import Node
        from tree_sitter_language_pack import get_parser
        from magika import Magika
    except ImportError:
        raise ImportError("One or more of the following dependencies are not installed: " +
                         "[ tree-sitter, tree-sitter-language-pack, magika ]" +
                         " Please install them using `pip install chonkie[code]`.")

  def _detect_language(self, bytes_text: bytes) -> str:
    """Detect the language of the code."""
    response = self.magika.identify_bytes(bytes_text)
    return response.output.label # type: ignore

  def _merge_extracted_nodes(self, extracted_nodes: List[Dict[str, Any]], text_bytes: bytes) -> Dict[str, Any]:
    """Merge the extracted nodes using byte positions."""
    if len(extracted_nodes) == 1:
      return extracted_nodes[0]

    first_node = extracted_nodes[0]
    last_node = extracted_nodes[-1]
    
    # Extract merged text using byte positions
    merged_bytes = text_bytes[first_node['start_byte']:last_node['end_byte']]
    merged_text = merged_bytes.decode('utf-8')
    
    return {
      "start_byte": first_node['start_byte'],
      "end_byte": last_node['end_byte'],
      "start_line": first_node['start_line'],
      "end_line": last_node['end_line'],
      "type": last_node['type'],
      "text": merged_text
    }

  def _extract_node(self, node: "Node") -> Dict[str, Any]:
    """Extract the node content."""
    text = node.text.decode() # type: ignore
    return {
      "text": text,
      "start_line": node.start_point[0],
      "end_line": node.end_point[0],
      "start_byte": node.start_byte,
      "end_byte": node.end_byte,
      "type": node.type,
    }

  def _handle_target_node_with_recursion(self, target_node: "Node", rule: SplitRule, text_bytes: bytes) -> List[Dict[str, Any]]:
    """Handle target node with potential recursive splitting."""
    if rule.recursive and self.chunk_size is not None:
      target_text = str(target_node.text.decode()) 
      target_token_count = self.tokenizer.count_tokens(target_text)
      
      if target_token_count > self.chunk_size:
        recursive_chunks = self._split_node(target_node, rule, text_bytes)
        if recursive_chunks:
          return recursive_chunks
    
    # Fallback: return as single node
    return [self._extract_node(target_node)]

  def _perform_sequential_splitting(self, all_children: List["Node"], target_indices: List[int], rule: SplitRule, text_bytes: bytes) -> List[Dict[str, Any]]:
    """Perform sequential splitting logic."""
    result_chunks = []
    start_idx = 0
    
    for target_idx in target_indices:
      # Create chunk from start_idx to target_idx (exclusive)
      if start_idx < target_idx:
        chunk_nodes = all_children[start_idx:target_idx]
        if chunk_nodes:
          chunk_exnodes = [self._extract_node(n) for n in chunk_nodes]
          merged_chunk = self._merge_extracted_nodes(chunk_exnodes, text_bytes)
          result_chunks.append(merged_chunk)
      
      # Handle target node with potential recursion
      target_node = all_children[target_idx]
      target_chunks = self._handle_target_node_with_recursion(target_node, rule, text_bytes)
      result_chunks.extend(target_chunks)
      
      start_idx = target_idx + 1
    
    # Handle remaining nodes after last target
    if start_idx < len(all_children):
      remaining_nodes = all_children[start_idx:]
      if remaining_nodes:
        remaining_exnodes = [self._extract_node(n) for n in remaining_nodes]
        merged_remaining = self._merge_extracted_nodes(remaining_exnodes, text_bytes)
        result_chunks.append(merged_remaining)
    
    return result_chunks

  def _split_node(self, node: "Node", rule: SplitRule, text_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract the split node with sequential splitting support (refactored)."""
    if isinstance(rule.body_child, str):
      if rule.body_child == "self":
        return [self._extract_node(node)]
      
      # Simple case: single-level child
      target_type = rule.body_child
      all_children = list(node.children)
      target_indices = [i for i, child in enumerate(all_children) if child.type == target_type]
      
      if not target_indices:
        return []
      
      return self._perform_sequential_splitting(all_children, target_indices, rule, text_bytes)
    
    else:
      # Complex case: path traversal through nested children
      current_node = node
      path = rule.body_child
      
      # Traverse to the final level
      for i, target_type in enumerate(path[:-1]):
        found_target = None
        for child in current_node.children:
          if child.type == target_type:
            found_target = child
            break
        
        if found_target is None:
          return []
        
        current_node = found_target
      
      # Handle final level
      final_target_type = path[-1]
      all_children = list(current_node.children)
      target_indices = [i for i, child in enumerate(all_children) if child.type == final_target_type]
      
      if not target_indices:
        return []
      
      return self._perform_sequential_splitting(all_children, target_indices, rule, text_bytes)


  def _extract_split_nodes(self, nodes: List["Node"], text_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract important information from the nodes."""
    exnodes: List[Dict[str, Any]] = []
    for node in nodes:
      # If node matches one with a split rule and the token count for it is larger then, split it.
      is_split = False
      if self.chunk_size is not None:
        for rule in self.language_config.split_rules:
            if node.type == rule.node_type:
              split_nodes = self._split_node(node, rule, text_bytes)
              if split_nodes:
                exnodes.extend(split_nodes)
              is_split = True
              break
            
      if not is_split:
        exnodes.append(self._extract_node(node))
    return exnodes

  def _should_merge_node_w_node_group(self, extracted_node: Dict[str, Any], extracted_node_group: List[Dict[str, Any]]) -> bool:
    """Check if the current node should be merged with the node group."""
    if not extracted_node_group:
      return False

    try:
      current_type = extracted_node['type']
      previous_type = extracted_node_group[-1]['type']
    except KeyError:
      print(extracted_node)
      print(extracted_node_group)

      raise KeyError 

    for rule in self.language_config.merge_rules:
      # First check if this is the bidirectional or not
      if rule.bidirectional and current_type in rule.node_types and previous_type in rule.node_types:
          return True
      elif not rule.bidirectional and previous_type in rule.node_types[0] and current_type in rule.node_types[1]:
            return True

    # If nothing matches, return false
    return False

  def _merge_extracted_nodes_by_type(self, exnodes: List[Dict[str, Any]], text_bytes: bytes) -> List[Dict[str, Any]]:
    """Merge the extracted nodes by type."""
    if len(exnodes) < 2:
      return exnodes

    merged_exnodes: List[Dict[str, Any]] = []
    current_group: List[Dict[str, Any]] = [exnodes[0]]
    i = 0
    while i < len(exnodes) - 1:
      current_exnode = exnodes[i+1]

      if self._should_merge_node_w_node_group(current_exnode, current_group):
        current_group.append(current_exnode)
      else:
        merged_exnodes.append(self._merge_extracted_nodes(current_group, text_bytes))
        current_group = [current_exnode]

      # Update the counter
      i += 1

    if current_group:
      merged_exnodes.append(self._merge_extracted_nodes(current_group, text_bytes))

    return merged_exnodes

  def _create_chunks_from_exnodes(self, exnodes: List[Dict[str, Any]], text_bytes: bytes) -> List[Chunk]:
    """Create chunks from the extracted nodes, avoiding small whitespace-only chunks."""
    chunks: List[Chunk] = []
    current_index = 0
    current_byte_pos = 0
    
    if not exnodes:
      original_text = text_bytes.decode('utf-8')
      token_count = self.tokenizer.count_tokens(original_text)
      return [Chunk(text=original_text, start_index=0, end_index=len(original_text), token_count=token_count)]
    
    # Sort by byte position
    exnodes.sort(key=lambda x: x['start_byte'])
    
    for i, exnode in enumerate(exnodes):
      # Check for gap before this node
      gap_text = ""
      if current_byte_pos < exnode['start_byte']:
        gap_bytes = text_bytes[current_byte_pos:exnode['start_byte']]
        gap_text = gap_bytes.decode('utf-8')
      
      # Get the main chunk text (extract from original text if needed)
      if 'text' in exnode:
        chunk_text = exnode['text']
      else:
        # Extract text from bytes using byte positions and decode properly
        chunk_bytes = text_bytes[exnode['start_byte']:exnode['end_byte']]
        chunk_text = chunk_bytes.decode('utf-8')
      
      # Track the chunk start position (before any gap merging)
      chunk_start_index = current_index
      
      # Decide whether to merge gap with current chunk or create separate chunks
      if gap_text:
        # Check if gap is small whitespace that should be merged
        if len(gap_text.strip()) == 0 and len(gap_text) <= 20:  # Small whitespace gap
          # Merge gap with current chunk - chunk will start from current_index (before gap)
          chunk_text = gap_text + chunk_text
          # Don't update current_index here - it stays where the chunk starts
        else:
          # Create separate chunk for gap if it's substantial
          token_count = self.tokenizer.count_tokens(gap_text)
          chunks.append(Chunk(
            text=gap_text,
            start_index=current_index,
            end_index=current_index + len(gap_text),
            token_count=token_count
          ))
          current_index += len(gap_text)
          chunk_start_index = current_index  # Update start for main chunk
      
      # Add the main chunk (possibly with merged gap)
      token_count = self.tokenizer.count_tokens(chunk_text)
      chunks.append(Chunk(
        text=chunk_text,
        start_index=chunk_start_index,
        end_index=chunk_start_index + len(chunk_text),
        token_count=token_count
      ))
      current_index = chunk_start_index + len(chunk_text)
      current_byte_pos = exnode['end_byte']
    
    # Add remaining text after last chunk
    if current_byte_pos < len(text_bytes):
      remaining_bytes = text_bytes[current_byte_pos:]
      remaining_text = remaining_bytes.decode('utf-8')
      if remaining_text:
        # Check if we should merge with previous chunk
        if (chunks and len(remaining_text.strip()) == 0 and len(remaining_text) <= 20):
          # Merge with last chunk
          last_chunk = chunks[-1]
          merged_text = last_chunk.text + remaining_text
          token_count = self.tokenizer.count_tokens(merged_text)
          chunks[-1] = Chunk(
            text=merged_text,
            start_index=last_chunk.start_index,
            end_index=last_chunk.end_index + len(remaining_text),
            token_count=token_count
          )
        else:
          # Create separate chunk
          token_count = self.tokenizer.count_tokens(remaining_text)
          chunks.append(Chunk(
            text=remaining_text,
            start_index=current_index,
            end_index=current_index + len(remaining_text),
            token_count=token_count
          ))
    
    return chunks

  def chunk(self, text: str) -> List[Chunk]:
    """Chunk the code."""
    # Encode text to bytes for consistent byte position handling
    text_bytes = text.encode('utf-8')
    
    # At this point, if the language is auto, we need to detect the language
    # and initialize the parser and language config
    if self.language == "auto":
        detected_language = self._detect_language(text_bytes)
        
        # Check if the detected language is supported
        if detected_language not in CodeLanguageRegistry:
            # If the detected language is not supported, fall back to a default
            # or raise an error with helpful message
            raise ValueError(f"Detected language '{detected_language}' is not supported. " +
                           f"Supported languages: {list(CodeLanguageRegistry.keys())}")
        
        # Initialize parser and language config for detected language
        self.parser = get_parser(detected_language) # type: ignore
        self.language_config = CodeLanguageRegistry[detected_language]
    
    # Create the tree-sitter tree
    tree = self.parser.parse(text_bytes) # type: ignore
    root = tree.root_node # type: ignore
    nodes = root.children # type: ignore

    # Extract and split the nodes
    exnodes = self._extract_split_nodes(nodes, text_bytes)

    # Merge the nodes based on type
    merged_exnodes = self._merge_extracted_nodes_by_type(exnodes, text_bytes)

    # return the final chunks
    chunks = self._create_chunks_from_exnodes(merged_exnodes, text_bytes)
    return chunks
