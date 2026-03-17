"""
Lightweight Pydantic-compatible base model using stdlib only.
Provides the same interface as Pydantic v2 BaseModel for environments
where pydantic is not installed.
"""
from __future__ import annotations
import copy
import inspect
from typing import Any, Dict, get_type_hints


class _ModelMeta(type):
    pass


class BaseModel(metaclass=_ModelMeta):
    """Minimal Pydantic-like BaseModel backed by __init__ + __post_init__."""

    model_config: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, **data):
        hints = {}
        for klass in reversed(type(self).__mro__):
            if klass is object:
                continue
            try:
                hints.update(get_type_hints(klass))
            except Exception:
                pass

        # Collect defaults from class attributes
        defaults = {}
        for klass in reversed(type(self).__mro__):
            for k, v in vars(klass).items():
                if k.startswith('_') or callable(v) or isinstance(v, (classmethod, staticmethod, property)):
                    continue
                if k in hints:
                    defaults[k] = copy.deepcopy(v)

        # Apply field defaults (Field objects)
        for k, v in defaults.items():
            if isinstance(v, _Field):
                if v.default is not _MISSING:
                    defaults[k] = v.default
                elif v.default_factory is not None:
                    defaults[k] = v.default_factory()
                else:
                    defaults.pop(k, None)

        # Merge provided data over defaults
        merged = {**defaults, **data}

        for key, val in merged.items():
            object.__setattr__(self, key, val)

        # Run validators
        self._run_validators()

    def _run_validators(self):
        """Run model_validator after and field_validators."""
        cls = type(self)
        # field validators
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if getattr(method, '_is_field_validator', False):
                field_name = method._validator_field
                if hasattr(self, field_name):
                    val = getattr(self, field_name)
                    result = method.__func__(cls, val)
                    object.__setattr__(self, field_name, result)

        # model validators
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if getattr(method, '_is_model_validator', False):
                result = method(self)
                if result is not None:
                    object.__setattr__(self, '_validated', True)

    def model_dump(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}

    def __repr__(self):
        fields = ', '.join(f'{k}={v!r}' for k, v in vars(self).items()
                           if not k.startswith('_'))
        return f"{type(self).__name__}({fields})"


_MISSING = object()


class _Field:
    def __init__(self, default=_MISSING, default_factory=None, ge=None, le=None, gt=None):
        self.default = default
        self.default_factory = default_factory
        self.ge = ge
        self.le = le
        self.gt = gt


def Field(default=_MISSING, *, default_factory=None, ge=None, le=None, gt=None):
    if default is _MISSING and default_factory is None:
        return _Field(default=_MISSING, default_factory=None, ge=ge, le=le, gt=gt)
    f = _Field(default=default, default_factory=default_factory, ge=ge, le=le, gt=gt)
    return f


def field_validator(field_name, *, mode="before"):
    """Decorator matching Pydantic v2 @field_validator."""
    def decorator(func):
        func._is_field_validator = True
        func._validator_field = field_name
        return classmethod(func)
    return decorator


def model_validator(*, mode="after"):
    """Decorator matching Pydantic v2 @model_validator."""
    def decorator(func):
        func._is_model_validator = True
        return func
    return decorator
