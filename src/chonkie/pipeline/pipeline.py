"""Core Pipeline class for chonkie."""

import inspect
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .registry import ComponentRegistry, ComponentType


class Pipeline:
    """A fluent API for building and executing chonkie pipelines.

    The Pipeline class provides a clean, chainable interface for processing
    documents through the CHOMP pipeline: CHef -> CHunker -> Refinery -> Porter/Handshake.

    Example:
        ```python
        from chonkie.pipeline import Pipeline

        # Simple pipeline
        chunks = (Pipeline()
            .fetch_from("file", path="document.txt")
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .execute())

        # Complex pipeline with refinement and export
        (Pipeline()
            .fetch_from("file", path="document.txt")
            .process_with("text")
            .chunk_with("recursive", chunk_size=512)
            .refine_with("overlap", merge_threshold=0.8)
            .export_with("json", output_path="chunks.json")
            .execute())
        ```

    """

    def __init__(self):
        """Initialize a new Pipeline."""
        self._steps = []
        self._data = None
        self._component_instances = {}  # Cache for component instances

    @classmethod
    def from_string(cls, pipeline_string: str, **global_params) -> "Pipeline":
        """Create a Pipeline from a string representation.
        
        Args:
            pipeline_string: String representation of the pipeline
            **global_params: Global parameters to apply to all components
            
        Returns:
            Pipeline instance configured according to the string
            
        Raises:
            ValueError: If the pipeline string format is invalid
            
        Examples:
            ```python
            # Simple pipeline
            pipeline = Pipeline.from_string("file -> text -> recursive")
            
            # With parameters
            pipeline = Pipeline.from_string(
                "file(path='doc.txt') -> text -> recursive(chunk_size=512) -> json(output_path='out.json')"
            )
            
            # With global parameters
            pipeline = Pipeline.from_string(
                "file -> text -> recursive -> overlap",
                chunk_size=512,
                show_progress=False
            )
            ```
        """
        pipeline = cls()
        
        # Clean and split the pipeline string
        pipeline_string = pipeline_string.strip()
        if not pipeline_string:
            raise ValueError("Pipeline string cannot be empty")
        
        # Split by -> and parse each component
        components = [comp.strip() for comp in pipeline_string.split("->")]
        
        for i, component_str in enumerate(components):
            try:
                component_type, component_name, params = pipeline._parse_component_string(component_str)
                
                # Merge global params with component-specific params
                merged_params = {**global_params, **params}
                
                # Add the appropriate step based on component type
                if component_type == ComponentType.FETCHER:
                    pipeline.fetch_from(component_name, **merged_params)
                elif component_type == ComponentType.CHEF:
                    pipeline.process_with(component_name, **merged_params)
                elif component_type == ComponentType.CHUNKER:
                    pipeline.chunk_with(component_name, **merged_params)
                elif component_type == ComponentType.REFINERY:
                    pipeline.refine_with(component_name, **merged_params)
                elif component_type == ComponentType.PORTER:
                    pipeline.export_with(component_name, **merged_params)
                elif component_type == ComponentType.HANDSHAKE:
                    pipeline.store_in(component_name, **merged_params)
                else:
                    raise ValueError(f"Unknown component type: {component_type}")
                    
            except Exception as e:
                raise ValueError(f"Error parsing component {i+1} '{component_str}': {e}") from e
        
        return pipeline

    def _parse_component_string(self, component_str: str) -> Tuple[ComponentType, str, Dict[str, Any]]:
        """Parse a single component string into type, name, and parameters.
        
        Args:
            component_str: Component string like "recursive(chunk_size=512)"
            
        Returns:
            Tuple of (component_type, component_name, parameters)
            
        Raises:
            ValueError: If component string format is invalid
        """
        # Match pattern: component_name(param1=value1, param2=value2)
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\((.*)\))?\s*$', component_str)
        
        if not match:
            raise ValueError(f"Invalid component format: '{component_str}'. Expected format: 'component_name' or 'component_name(param=value)'")
        
        component_name = match.group(1)
        params_str = match.group(2) or ""
        
        # Parse parameters
        params = {}
        if params_str.strip():
            params = self._parse_parameters(params_str)
        
        # Determine component type by looking it up in registry
        try:
            component_info = ComponentRegistry.get_component(component_name)
            return component_info.component_type, component_name, params
        except ValueError:
            # If not found, provide helpful error with available components
            available = ComponentRegistry.get_aliases()
            raise ValueError(
                f"Unknown component '{component_name}'. Available components: {sorted(available)}"
            )

    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """Parse parameter string into a dictionary.
        
        Args:
            params_str: Parameter string like "chunk_size=512, show_progress=False"
            
        Returns:
            Dictionary of parsed parameters
            
        Raises:
            ValueError: If parameter format is invalid
        """
        params = {}
        
        # Split by comma, but be careful with nested structures
        param_parts = []
        current_part = ""
        paren_depth = 0
        bracket_depth = 0
        in_quotes = False
        quote_char = None
        
        for char in params_str:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif not in_quotes:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                elif char == ',' and paren_depth == 0 and bracket_depth == 0:
                    param_parts.append(current_part.strip())
                    current_part = ""
                    continue
            
            current_part += char
        
        if current_part.strip():
            param_parts.append(current_part.strip())
        
        # Parse each parameter
        for param_part in param_parts:
            if '=' not in param_part:
                raise ValueError(f"Invalid parameter format: '{param_part}'. Expected 'key=value'")
            
            key, value_str = param_part.split('=', 1)
            key = key.strip()
            value_str = value_str.strip()
            
            # Parse the value
            try:
                value = self._parse_value(value_str)
                params[key] = value
            except Exception as e:
                raise ValueError(f"Error parsing value for parameter '{key}': {e}") from e
        
        return params

    def _parse_value(self, value_str: str) -> Any:
        """Parse a string value into appropriate Python type.
        
        Args:
            value_str: String representation of the value
            
        Returns:
            Parsed value with appropriate type
        """
        value_str = value_str.strip()
        
        # Handle None
        if value_str.lower() == 'none':
            return None
        
        # Handle booleans
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False
        
        # Handle strings (quoted)
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]  # Remove quotes
        
        # Handle lists
        if value_str.startswith('[') and value_str.endswith(']'):
            list_content = value_str[1:-1].strip()
            if not list_content:
                return []
            
            # Split list items (simple comma split for now)
            items = [item.strip() for item in list_content.split(',')]
            return [self._parse_value(item) for item in items]
        
        # Handle numbers
        try:
            # Try integer first
            if '.' not in value_str:
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            pass
        
        # Default to string (unquoted)
        return value_str

    @lru_cache(maxsize=128)
    def _get_parameter_split(self, component_class) -> Tuple[Set[str], Set[str]]:
        """Get parameter names for __init__ and __call__ methods.
        
        Args:
            component_class: The component class to inspect
            
        Returns:
            Tuple of (init_params, call_params) parameter name sets
        """
        try:
            # Get method signatures
            init_sig = inspect.signature(component_class.__init__)
            call_sig = inspect.signature(component_class.__call__)
            
            # Extract parameter names (excluding 'self' and common params)
            init_params = set(init_sig.parameters.keys()) - {"self"}
            call_params = set(call_sig.parameters.keys()) - {"self", "text"}
            
            return init_params, call_params
        except Exception as e:
            # Fallback: assume all params go to __init__ if inspection fails
            return set(), set()

    def _split_parameters(self, component_class, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split kwargs into init and call parameters based on method signatures.
        
        Args:
            component_class: The component class to inspect
            kwargs: All parameters provided by user
            
        Returns:
            Tuple of (init_kwargs, call_kwargs)
            
        Raises:
            ValueError: If unknown parameters are provided
        """
        init_params, call_params = self._get_parameter_split(component_class)
        
        # Split kwargs based on which method accepts each parameter
        init_kwargs = {}
        call_kwargs = {}
        unknown_params = []
        
        for key, value in kwargs.items():
            if key in init_params:
                init_kwargs[key] = value
            elif key in call_params:
                call_kwargs[key] = value
            else:
                # Handle parameter conflicts (exists in both) - prefer __init__ for config-like params
                config_like_params = {
                    "chunk_size", "model_name", "api_key", "tokenizer", "tokenizer_or_token_counter",
                    "rules", "min_characters_per_chunk", "merge_threshold", "context_size",
                    "embedding_model", "similarity_threshold", "max_chunk_size"
                }
                
                if key in config_like_params:
                    init_kwargs[key] = value
                elif key in {"show_progress", "batch_size", "verbose"}:
                    call_kwargs[key] = value
                else:
                    unknown_params.append(key)
        
        if unknown_params:
            available_params = sorted(init_params | call_params)
            raise ValueError(
                f"Unknown parameters for {component_class.__name__}: {unknown_params}. "
                f"Available parameters: {available_params}"
            )
        
        return init_kwargs, call_kwargs

    def fetch_from(self, source_type: str, **kwargs) -> "Pipeline":
        """Fetch data from a source.

        Args:
            source_type: Type of source fetcher to use (e.g., "file")
            **kwargs: Arguments passed to the fetcher component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If source_type is not a registered fetcher

        Example:
            ```python
            pipeline.fetch_from("file", path="document.txt")
            ```

        """
        component = ComponentRegistry.get_fetcher(source_type)
        self._steps.append({"type": "fetch", "component": component, "kwargs": kwargs})
        return self

    def process_with(self, chef_type: str, **kwargs) -> "Pipeline":
        """Process data with a chef component.

        Args:
            chef_type: Type of chef to use (e.g., "text")
            **kwargs: Arguments passed to the chef component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If chef_type is not a registered chef

        Example:
            ```python
            pipeline.process_with("text", clean_whitespace=True)
            ```

        """
        component = ComponentRegistry.get_chef(chef_type)
        self._steps.append({
            "type": "process",
            "component": component,
            "kwargs": kwargs,
        })
        return self

    def chunk_with(self, chunker_type: str, **kwargs) -> "Pipeline":
        """Chunk data with a chunker component.

        Args:
            chunker_type: Type of chunker to use (e.g., "recursive", "semantic")
            **kwargs: Arguments passed to the chunker component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If chunker_type is not a registered chunker

        Example:
            ```python
            pipeline.chunk_with("recursive", chunk_size=512, chunk_overlap=50)
            ```

        """
        component = ComponentRegistry.get_chunker(chunker_type)
        self._steps.append({"type": "chunk", "component": component, "kwargs": kwargs})
        return self

    def refine_with(self, refinery_type: str, **kwargs) -> "Pipeline":
        """Refine chunks with a refinery component.

        Args:
            refinery_type: Type of refinery to use (e.g., "overlap", "embedding")
            **kwargs: Arguments passed to the refinery component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If refinery_type is not a registered refinery

        Example:
            ```python
            pipeline.refine_with("overlap", merge_threshold=0.8)
            ```

        """
        component = ComponentRegistry.get_refinery(refinery_type)
        self._steps.append({"type": "refine", "component": component, "kwargs": kwargs})
        return self

    def export_with(self, porter_type: str, **kwargs) -> "Pipeline":
        """Export chunks with a porter component.

        Args:
            porter_type: Type of porter to use (e.g., "json", "datasets")
            **kwargs: Arguments passed to the porter component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If porter_type is not a registered porter

        Example:
            ```python
            pipeline.export_with("json", output_path="chunks.json")
            ```

        """
        component = ComponentRegistry.get_porter(porter_type)
        self._steps.append({"type": "export", "component": component, "kwargs": kwargs})
        return self

    def store_in(self, handshake_type: str, **kwargs) -> "Pipeline":
        """Store chunks in a vector database with a handshake component.

        Args:
            handshake_type: Type of handshake to use (e.g., "chroma", "qdrant")
            **kwargs: Arguments passed to the handshake component

        Returns:
            Pipeline instance for method chaining

        Raises:
            ValueError: If handshake_type is not a registered handshake

        Example:
            ```python
            pipeline.store_in("chroma", collection_name="documents")
            ```

        """
        component = ComponentRegistry.get_handshake(handshake_type)
        self._steps.append({"type": "store", "component": component, "kwargs": kwargs})
        return self

    def execute(self, texts: Optional[Union[str, List[str]]] = None) -> Any:
        """Execute the pipeline and return the final result.
        
        The pipeline automatically reorders steps according to the CHOMP flow:
        Fetcher -> Chef -> Chunker -> Refinery(ies) -> Porter/Handshake
        
        This allows components to be defined in any order during pipeline building,
        but ensures correct execution order.

        Args:
            texts: Optional text input. Can be a single string or list of strings.
                   When provided, the fetcher step becomes optional.

        Returns:
            The output of the final pipeline step

        Raises:
            ValueError: If pipeline has no steps or invalid step configuration
            RuntimeError: If pipeline execution fails

        Examples:
            ```python
            # Traditional fetcher-based pipeline
            pipeline = (Pipeline()
                .fetch_from("file", path="doc.txt")
                .process_with("text")
                .chunk_with("recursive", chunk_size=512))
            chunks = pipeline.execute()
            
            # Direct text input (fetcher optional)
            pipeline = (Pipeline()
                .process_with("text")
                .chunk_with("recursive", chunk_size=512))
            chunks = pipeline.execute(texts="Hello world")
            
            # Multiple texts
            chunks = pipeline.execute(texts=["Text 1", "Text 2", "Text 3"])
            ```

        """
        if not self._steps:
            raise ValueError("Pipeline has no steps to execute")

        # Reorder steps according to CHOMP pipeline flow
        ordered_steps = self._reorder_steps()
        
        # Validate the pipeline configuration (considering text input)
        self._validate_pipeline(ordered_steps, has_text_input=(texts is not None))

        # Initialize data based on input
        if texts is not None:
            # Direct text input - normalize to list format for consistent processing
            if isinstance(texts, str):
                data = [texts]
            else:
                data = texts
        else:
            data = None

        # Execute pipeline steps
        for i, step in enumerate(ordered_steps):
            try:
                # Skip fetcher step if we have direct text input
                if texts is not None and step["type"] == "fetch":
                    continue
                    
                data = self._execute_step(step, data)
            except Exception as e:
                step_info = f"step {i + 1} ({step['type']})"
                raise RuntimeError(f"Pipeline failed at {step_info}: {e}") from e

        return data

    def _reorder_steps(self) -> List[Dict[str, Any]]:
        """Reorder pipeline steps according to CHOMP flow.
        
        Returns:
            List of steps in correct execution order
        """
        # Define the correct order of component types
        type_order = {
            "fetch": 0,
            "process": 1, 
            "chunk": 2,
            "refine": 3,
            "export": 4,
            "store": 5
        }
        
        # Group steps by type
        steps_by_type = {}
        for step in self._steps:
            step_type = step["type"]
            if step_type not in steps_by_type:
                steps_by_type[step_type] = []
            steps_by_type[step_type].append(step)
        
        # Build ordered list
        ordered_steps = []
        
        # Add steps in the correct order
        for step_type in sorted(type_order.keys(), key=lambda x: type_order[x]):
            if step_type in steps_by_type:
                if step_type == "refine":
                    # For refineries, maintain the order they were added
                    ordered_steps.extend(steps_by_type[step_type])
                else:
                    # For other types, there should typically be only one
                    # If multiple exist, use the last one defined (most recent)
                    ordered_steps.append(steps_by_type[step_type][-1])
        
        return ordered_steps

    def _validate_pipeline(self, ordered_steps: List[Dict[str, Any]], has_text_input: bool = False) -> None:
        """Validate that the pipeline configuration is valid.
        
        Args:
            ordered_steps: Steps in execution order
            has_text_input: Whether direct text input is provided to execute()
            
        Raises:
            ValueError: If pipeline configuration is invalid
        """
        if not ordered_steps:
            raise ValueError("Pipeline has no steps to execute")
        
        step_types = [step["type"] for step in ordered_steps]
        
        # Check that we have at least a chunker (minimum viable pipeline)
        if "chunk" not in step_types:
            raise ValueError("Pipeline must include a chunker component (use chunk_with())")
        
        # Check fetcher requirements based on input method
        if not has_text_input and "fetch" not in step_types:
            raise ValueError(
                "Pipeline must include a fetcher component (use fetch_from()) "
                "or provide text input to execute(texts=...)"
            )
        
        if has_text_input and "fetch" in step_types:
            # This is okay - fetcher will be skipped when text input is provided
            pass
        
        # Warn about common issues (but don't fail)
        if "process" not in step_types:
            # This is okay - some pipelines might not need text processing
            pass
        
        # Check for conflicting export/store operations
        has_export = "export" in step_types
        has_store = "store" in step_types
        
        if has_export and has_store:
            # This is actually okay - user might want both
            pass

    def _execute_step(self, step: Dict[str, Any], input_data: Any) -> Any:
        """Execute a single pipeline step.

        Args:
            step: Step configuration dictionary
            input_data: Input data from previous step

        Returns:
            Output data from this step

        """
        component_info = step["component"]
        kwargs = step["kwargs"]
        step_type = step["type"]

        # Auto-detect parameter separation
        try:
            init_kwargs, call_kwargs = self._split_parameters(
                component_info.component_class, kwargs
            )
        except Exception as e:
            raise ValueError(
                f"Parameter analysis failed for {component_info.component_class.__name__}: {e}"
            ) from e

        # Create component instance with init parameters only
        component_key = (component_info.name, tuple(sorted(init_kwargs.items())))
        if component_key not in self._component_instances:
            try:
                self._component_instances[component_key] = component_info.component_class(**init_kwargs)
            except Exception as e:
                raise ValueError(
                    f"Failed to create {component_info.component_class.__name__} with parameters {init_kwargs}: {e}"
                ) from e

        component_instance = self._component_instances[component_key]

        # Use __call__ method with call parameters
        try:
            if step_type == "fetch":
                # For fetch steps, we pass call_kwargs to __call__, not input_data
                return component_instance(**call_kwargs)
            else:
                # For other steps, pass input_data and call_kwargs to __call__
                if call_kwargs:
                    return component_instance(input_data, **call_kwargs)
                else:
                    return component_instance(input_data)
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute {component_info.component_class.__name__}.__call__() "
                f"with call parameters {call_kwargs}: {e}"
            ) from e

    def reset(self) -> "Pipeline":
        """Reset the pipeline to its initial state.

        Returns:
            Pipeline instance for method chaining

        """
        self._steps.clear()
        self._data = None
        self._component_instances.clear()
        return self

    def copy(self) -> "Pipeline":
        """Create a copy of the current pipeline.

        Returns:
            New Pipeline instance with the same steps

        """
        new_pipeline = Pipeline()
        new_pipeline._steps = self._steps.copy()
        return new_pipeline

    def describe(self) -> str:
        """Get a human-readable description of the pipeline.

        Returns:
            String description of the pipeline steps

        """
        if not self._steps:
            return "Empty pipeline"

        descriptions = []
        for step in self._steps:
            component = step["component"]
            step_type = step["type"]
            descriptions.append(f"{step_type}({component.alias})")

        return " -> ".join(descriptions)

    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        return f"Pipeline({self.describe()})"
