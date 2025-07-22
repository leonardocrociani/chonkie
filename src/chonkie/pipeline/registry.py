"""Component registry for pipeline components."""

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, get_type_hints


class ComponentType(Enum):
    """Types of pipeline components."""

    FETCHER = "fetcher"
    CHEF = "chef"
    CHUNKER = "chunker"
    REFINERY = "refinery"
    PORTER = "porter"
    HANDSHAKE = "handshake"


@dataclass
class Component:
    """Metadata about a pipeline component."""

    name: str
    alias: str
    component_class: Type
    component_type: ComponentType
    input_type: Type = Any
    output_type: Type = Any
    default_args: Optional[Dict[str, Any]] = None
    required_args: Optional[List[str]] = None

    def __post_init__(self):
        """Validate component after creation."""
        if not self.name:
            raise ValueError("Component name cannot be empty")
        if not self.alias:
            raise ValueError("Component alias cannot be empty")
        if not self.component_class:
            raise ValueError("Component class cannot be None")


class _ComponentRegistry:
    """Internal component registry class."""

    def __init__(self):
        self._components: Dict[str, Component] = {}
        self._aliases: Dict[str, str] = {}  # alias -> name mapping
        self._component_types: Dict[ComponentType, List[str]] = {
            ct: [] for ct in ComponentType
        }
        self._initialized = False

    def register(
        self,
        name: str,
        alias: str,
        component_class: Type,
        component_type: ComponentType,
        input_type: Type = Any,
        output_type: Type = Any,
        default_args: Optional[Dict[str, Any]] = None,
        required_args: Optional[List[str]] = None,
    ) -> None:
        """Register a component in the registry.

        Args:
            name: Full name of the component (usually class name)
            alias: Short alias for the component (used in string pipelines)
            component_class: The actual component class
            component_type: Type of component (fetcher, chunker, etc.)
            input_type: Expected input type
            output_type: Expected output type
            default_args: Default arguments for the component
            required_args: Required arguments for the component

        Raises:
            ValueError: If component name/alias conflicts exist

        """
        # Check for name conflicts
        if name in self._components:
            existing = self._components[name]
            if existing.component_class is component_class:
                # Same class, same registration - this is fine (idempotent)
                return
            else:
                raise ValueError(
                    f"Component name '{name}' already registered with different class"
                )

        # Check for alias conflicts
        if alias in self._aliases:
            existing_name = self._aliases[alias]
            if existing_name != name:
                raise ValueError(
                    f"Alias '{alias}' already used by component '{existing_name}'"
                )

        # Create component info
        info = Component(
            name=name,
            alias=alias,
            component_class=component_class,
            component_type=component_type,
            input_type=input_type,
            output_type=output_type,
            default_args=default_args or {},
            required_args=required_args or [],
        )

        # Register the component
        self._components[name] = info
        self._aliases[alias] = name
        self._component_types[component_type].append(name)

    def get_component(self, name_or_alias: str) -> Component:
        """Get component info by name or alias.

        Args:
            name_or_alias: Component name or alias

        Returns:
            Component for the requested component

        Raises:
            ValueError: If component is not found

        """
        # Try alias first, then name
        if name_or_alias in self._aliases:
            name = self._aliases[name_or_alias]
        else:
            name = name_or_alias

        if name not in self._components:
            available_aliases = list(self._aliases.keys())
            raise ValueError(
                f"Unknown component: '{name_or_alias}'. "
                f"Available components: {available_aliases}"
            )

        return self._components[name]

    def list_components(
        self, component_type: Optional[ComponentType] = None
    ) -> List[Component]:
        """List all registered components, optionally filtered by type.

        Args:
            component_type: Optional filter by component type

        Returns:
            List of Component objects

        """
        if component_type:
            names = self._component_types[component_type]
            return [self._components[name] for name in names]
        return list(self._components.values())

    def get_aliases(self, component_type: Optional[ComponentType] = None) -> List[str]:
        """Get all available aliases, optionally filtered by type.

        Args:
            component_type: Optional filter by component type

        Returns:
            List of component aliases

        """
        if component_type:
            names = self._component_types[component_type]
            return [self._components[name].alias for name in names]
        return list(self._aliases.keys())

    def get_fetcher(self, alias: str) -> Component:
        """Get a fetcher component by alias.

        Args:
            alias: Fetcher alias

        Returns:
            Component info for the fetcher

        Raises:
            ValueError: If fetcher not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.FETCHER:
            raise ValueError(f"'{alias}' is not a fetcher component")
        return component

    def get_chef(self, alias: str) -> Component:
        """Get a chef component by alias.

        Args:
            alias: Chef alias

        Returns:
            Component info for the chef

        Raises:
            ValueError: If chef not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.CHEF:
            raise ValueError(f"'{alias}' is not a chef component")
        return component

    def get_chunker(self, alias: str) -> Component:
        """Get a chunker component by alias.

        Args:
            alias: Chunker alias

        Returns:
            Component info for the chunker

        Raises:
            ValueError: If chunker not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.CHUNKER:
            raise ValueError(f"'{alias}' is not a chunker component")
        return component

    def get_refinery(self, alias: str) -> Component:
        """Get a refinery component by alias.

        Args:
            alias: Refinery alias

        Returns:
            Component info for the refinery

        Raises:
            ValueError: If refinery not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.REFINERY:
            raise ValueError(f"'{alias}' is not a refinery component")
        return component

    def get_porter(self, alias: str) -> Component:
        """Get a porter component by alias.

        Args:
            alias: Porter alias

        Returns:
            Component info for the porter

        Raises:
            ValueError: If porter not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.PORTER:
            raise ValueError(f"'{alias}' is not a porter component")
        return component

    def get_handshake(self, alias: str) -> Component:
        """Get a handshake component by alias.

        Args:
            alias: Handshake alias

        Returns:
            Component info for the handshake

        Raises:
            ValueError: If handshake not found

        """
        component = self.get_component(alias)
        if component.component_type != ComponentType.HANDSHAKE:
            raise ValueError(f"'{alias}' is not a handshake component")
        return component

    def is_registered(self, name_or_alias: str) -> bool:
        """Check if a component is registered.

        Args:
            name_or_alias: Component name or alias

        Returns:
            True if component is registered, False otherwise

        """
        return name_or_alias in self._aliases or name_or_alias in self._components

    def unregister(self, name_or_alias: str) -> None:
        """Unregister a component (mainly for testing).

        Args:
            name_or_alias: Component name or alias to unregister

        """
        if name_or_alias in self._aliases:
            name = self._aliases[name_or_alias]
            alias = name_or_alias
        elif name_or_alias in self._components:
            name = name_or_alias
            alias = self._components[name].alias
        else:
            return  # Component not registered

        # Remove from all tracking structures
        component_info = self._components[name]
        component_type = component_info.component_type

        del self._components[name]
        del self._aliases[alias]
        self._component_types[component_type].remove(name)

    def clear(self) -> None:
        """Clear all registered components (mainly for testing)."""
        self._components.clear()
        self._aliases.clear()
        for component_list in self._component_types.values():
            component_list.clear()


def _infer_types(cls: Type, component_type: ComponentType) -> tuple[Type, Type]:
    """Infer input/output types from method signatures.

    Args:
        cls: Component class
        component_type: Type of component

    Returns:
        Tuple of (input_type, output_type)

    """
    method_map = {
        ComponentType.FETCHER: "fetch",
        ComponentType.CHEF: "process",
        ComponentType.CHUNKER: "chunk",
        ComponentType.REFINERY: "refine",
        ComponentType.PORTER: "export",
        ComponentType.HANDSHAKE: "write",
    }

    method_name = method_map.get(component_type)
    if not method_name or not hasattr(cls, method_name):
        return Any, Any

    try:
        method = getattr(cls, method_name)
        type_hints = get_type_hints(method)

        # Get input type from first parameter (after self)
        sig = inspect.signature(method)
        params = list(sig.parameters.values())[1:]  # Skip 'self'
        input_type = type_hints.get(params[0].name, Any) if params else Any

        # Get output type from return annotation
        output_type = type_hints.get("return", Any)

        return input_type, output_type
    except Exception:
        return Any, Any


def _infer_args(cls: Type) -> tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
    """Infer default and required args from __init__ signature.

    Args:
        cls: Component class

    Returns:
        Tuple of (default_args, required_args)

    """
    try:
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())[1:]  # Skip 'self'

        default_args = {}
        required_args = []

        for param in params:
            # Skip common base class parameters that shouldn't be exposed
            if param.name in ["tokenizer_or_token_counter", "tokenizer"]:
                continue

            if param.default != inspect.Parameter.empty:
                default_args[param.name] = param.default
            else:
                required_args.append(param.name)

        return (
            default_args if default_args else None,
            required_args if required_args else None,
        )
    except Exception:
        return None, None


def pipeline_component(
    alias: str,
    component_type: ComponentType,
    input_type: Optional[Type] = None,
    output_type: Optional[Type] = None,
    default_args: Optional[Dict[str, Any]] = None,
    required_args: Optional[List[str]] = None,
    auto_infer_types: bool = True,
    auto_infer_args: bool = True,
):
    """Class Decorator that registers a class as a pipeline component.

    Args:
        alias: Short name for the component (used in string pipelines)
        component_type: Type of component (fetcher, chunker, etc.)
        input_type: Expected input type (auto-inferred if None and auto_infer_types=True)
        output_type: Expected output type (auto-inferred if None and auto_infer_types=True)
        default_args: Default arguments for the component (auto-inferred if None and auto_infer_args=True)
        required_args: Required arguments for the component (auto-inferred if None and auto_infer_args=True)
        auto_infer_types: Whether to auto-infer types from method signatures
        auto_infer_args: Whether to auto-infer args from __init__ signature

    Returns:
        Decorator function

    Example:
        @pipeline_component("RecursiveChunker", ComponentType.CHUNKER)
        class RecursiveChunker(BaseChunker):
            pass

    """

    def decorator(cls):
        # Validate that the class has required methods
        required_methods = {
            ComponentType.FETCHER: ["fetch"],
            ComponentType.CHEF: ["process"],
            ComponentType.CHUNKER: ["chunk"],
            ComponentType.REFINERY: ["refine"],
            ComponentType.PORTER: ["export"],
            ComponentType.HANDSHAKE: ["write"],
        }

        required = required_methods.get(component_type, [])
        for method_name in required:
            if not hasattr(cls, method_name):
                raise ValueError(
                    f"{cls.__name__} must implement {method_name}() method "
                    f"to be registered as {component_type.value}"
                )

        # Auto-infer types if requested and not provided
        final_input_type = input_type
        final_output_type = output_type

        if auto_infer_types and (input_type is None or output_type is None):
            inferred_input, inferred_output = _infer_types(cls, component_type)
            final_input_type = final_input_type or inferred_input
            final_output_type = final_output_type or inferred_output

        # Auto-infer args if requested and not provided
        final_default_args = default_args
        final_required_args = required_args

        if auto_infer_args and (default_args is None or required_args is None):
            inferred_defaults, inferred_required = _infer_args(cls)
            final_default_args = final_default_args or inferred_defaults
            final_required_args = final_required_args or inferred_required

        # Register the component
        ComponentRegistry.register(
            name=cls.__name__,
            alias=alias,
            component_class=cls,
            component_type=component_type,
            input_type=final_input_type or Any,
            output_type=final_output_type or Any,
            default_args=final_default_args,
            required_args=final_required_args,
        )

        # Add metadata to the class for introspection
        cls._pipeline_component_info = {
            "alias": alias,
            "component_type": component_type,
            "input_type": final_input_type,
            "output_type": final_output_type,
            "default_args": final_default_args,
            "required_args": final_required_args,
        }

        return cls

    return decorator


# Specialized decorators for each component type
def fetcher(alias: str, **kwargs):
    """ClassDecorator for fetcher components."""
    return pipeline_component(
        alias=alias, component_type=ComponentType.FETCHER, **kwargs
    )


def chef(alias: str, **kwargs):
    """Class decorator for chef components."""
    return pipeline_component(alias=alias, component_type=ComponentType.CHEF, **kwargs)


def chunker(alias: str, **kwargs):
    """Class decorator for chunker components."""
    return pipeline_component(
        alias=alias, component_type=ComponentType.CHUNKER, **kwargs
    )


def refinery(alias: str, **kwargs):
    """Class decorator for refinery components."""
    return pipeline_component(
        alias=alias, component_type=ComponentType.REFINERY, **kwargs
    )


def porter(alias: str, **kwargs):
    """Class decorator for porter components."""
    return pipeline_component(
        alias=alias, component_type=ComponentType.PORTER, **kwargs
    )


def handshake(alias: str, **kwargs):
    """Class decorator for handshake components."""
    return pipeline_component(
        alias=alias, component_type=ComponentType.HANDSHAKE, **kwargs
    )


# Global registry instance
ComponentRegistry = _ComponentRegistry()
