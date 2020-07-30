import abc
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
import numpy  # type: ignore
from typing import Any, Tuple, Iterator, Callable, MutableMapping

from .dimension import Dimension, Scalar


class Scale(abc.ABC):
    @abc.abstractmethod
    def unit(self, dimension: Dimension) -> Tuple["Quantity", str]: ...


def _scalar_units(ufunc: numpy.ufunc, *inputs: "Quantity", offset: int = 0) -> Tuple[Dimension, Scale]:
    dimension = Scalar
    scale = inputs[0].scale
    for i, x in enumerate(inputs):
        if x.dimension != dimension:
            raise ValueError(
                "Invalid mass dimension for argument {} to {}: got {}, expected {}"
                .format(i + offset, ufunc.__name__, x.dimension, dimension)
            )

    return dimension, scale


def _match_units(ufunc: numpy.ufunc, *inputs: "Quantity", offset: int = 0) -> Tuple[Dimension, Scale]:
    dimension = inputs[0].dimension
    scale = inputs[0].scale
    for i, x in enumerate(inputs):
        if x.dimension != dimension:
            raise ValueError(
                "Invalid mass dimension for argument {} to {}: got {}, expected {}"
                .format(i + offset, ufunc.__name__, x.dimension, dimension)
            )
        if x.dimension != Scalar and x.scale != scale:
            raise ValueError(
                "Mismatched scale for argument {} to {}"
                .format(i + offset, ufunc.__name__)
            )

    return dimension, scale


def _multiply_units(ufunc: numpy.ufunc, *inputs: "Quantity", offset: int = 0) -> Tuple[Dimension, Scale]:
    dimension = inputs[0].dimension
    scale = inputs[0].scale
    for i, x in enumerate(inputs):
        if i > 0:
            if dimension == Scalar and x.dimension != Scalar:
                scale = x.scale
            dimension = dimension * x.dimension
            if x.dimension != Scalar and x.scale != scale:
                raise ValueError(
                    "Mismatched scale for argument {} to {}"
                    .format(i + offset, ufunc.__name__)
                )

    return dimension, scale


def _divide_units(ufunc: numpy.ufunc, *inputs: "Quantity", offset: int = 0) -> Tuple[Dimension, Scale]:
    dimension = inputs[0].dimension
    scale = inputs[0].scale
    for i, x in enumerate(inputs):
        if i > 0:
            if dimension == Scalar and x.dimension != Scalar:
                scale = x.scale
            dimension = dimension / x.dimension
            if x.dimension != Scalar and x.scale != scale:
                raise ValueError(
                    "Mismatched scale for argument {} to {}"
                    .format(i + offset, ufunc.__name__)
                )

    return dimension, scale


def _power_units(ufunc: numpy.ufunc, *inputs: "Quantity", offset: int = 0) -> Tuple[Dimension, Scale]:
    dimension = inputs[0].dimension
    scale = inputs[0].scale
    for i, x in enumerate(inputs):
        if i > 0:
            if x.dimension != Scalar:
                raise ValueError(
                    "Invalid mass dimension for argument {} to {}: got {}, expected {}"
                    .format(i + offset, ufunc.__name__, x.dimension, Scalar)
                )
            dimension = dimension**x.value

    return dimension, scale


@dataclass(frozen=True)
class _UfuncUnits:
    unit_map: Callable[..., Tuple[Dimension, Scale]] = _scalar_units
    wrap_output: bool = True


_ufuncs: MutableMapping[numpy.ufunc, _UfuncUnits] = defaultdict(_UfuncUnits)


def _op(method: Callable[["Quantity", Any], Any]) -> Callable[["Quantity", Any], "Quantity"]:
    @wraps(method)
    def fn(self: "Quantity", other: Any) -> "Quantity":
        result: Quantity = getattr(super(Quantity, self), method.__name__)(other)
        return result
    return fn


def _iop(method: Callable[["Quantity", Any], Any]) -> Callable[["Quantity", Any], Any]:
    @wraps(method)
    def fn(self: "Quantity", other: Any) -> Any:
        if isinstance(self.value, numpy.ndarray):
            return getattr(super(Quantity, self), method.__name__)(other)
        else:
            return NotImplemented
    return fn


@dataclass(frozen=True, eq=False, order=False)
class Quantity(numpy.lib.mixins.NDArrayOperatorsMixin):
    value: Any
    dimension: Dimension
    scale: Scale

    def set_scale(self, curr: "Quantity", new: "Quantity") -> "Quantity":
        if self.scale != curr.scale:
            raise ValueError(
                "Error setting scale: Can't set scale for {} with {}"
                .format(self, curr)
            )
        if curr.dimension != new.dimension:
            raise ValueError(
                "Error setting scale: {} and {} have different mass dimensions"
                .format(curr, new)
            )
        if curr.scale == new.scale:
            raise ValueError(
                "Error setting scale: Initial and final scale are the same"
            )
        return Quantity(
            self.value * self.dimension.scale(
                (new.value / curr.value)**(1 / curr.dimension.mass_dim)
            ),
            self.dimension,
            new.scale,
        )

    def in_unit(self, val: "Quantity") -> Any:
        try:
            res = self / val
        except ValueError:
            raise ValueError("Can't convert units: incompatible scales")
        if res.dimension.mass_dim != 0:
            raise ValueError(
                "Can't convert units: incompatible mass dimensions {} and {}"
                .format(self.dimension.mass_dim, val.dimension.mass_dim)
            )
        return res.value

    def __str__(self) -> str:
        unit, suffix = self.scale.unit(self.dimension)
        if suffix != "":
            return str(self.in_unit(unit)) + " " + suffix
        else:
            return str(self.in_unit(unit))

    def __format__(self, format_str: str) -> str:
        unit, suffix = self.scale.unit(self.dimension)
        if suffix != "":
            return ("{:" + format_str + "} {}").format(
                self.in_unit(unit),
                suffix,
            )
        else:
            return ("{:" + format_str + "}").format(
                self.in_unit(unit),
            )

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, key: Any) -> "Quantity":
        return Quantity(
            self.value[key],
            self.dimension,
            self.scale,
        )

    def __iter__(self) -> "Iterator[Quantity]":
        for value in iter(self.value):
            yield Quantity(
                value,
                self.dimension,
                self.scale,
            )

    def __reversed__(self) -> "Iterator[Quantity]":
        for value in reversed(self.value):
            yield Quantity(
                value,
                self.dimension,
                self.scale,
            )

    def __array_ufunc__(self, ufunc: numpy.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        ufunc_units = _ufuncs[ufunc]
        result_dimension, result_scale = ufunc_units.unit_map(  # type: ignore
            ufunc,
            *(
                i if isinstance(i, Quantity) else Quantity(i, Scalar, self.scale)
                for i in inputs
            ),
        )

        inputs = tuple(x.value if isinstance(x, Quantity) else x
                       for x in inputs)

        out = kwargs.get('out', ())
        if out:
            if ufunc_units.wrap_output:
                for x in out:
                    if not isinstance(x, Quantity):
                        raise ValueError(
                            "Invalid type for '{}' output: should be Quantity"
                            .format(ufunc.__name__)
                        )
                    if x.dimension != result_dimension:
                        raise ValueError(
                            "Invalid mass dimension for '{}' output: was {}, should be {}"
                            .format(ufunc.__name__, x.dimension, result_dimension)
                        )
                    if x.scale != result_scale:
                        raise ValueError(
                            "Mismatched scales for '{}' output"
                            .format(ufunc.__name__)
                        )
            else:
                for x in out:
                    if isinstance(x, Quantity):
                        raise ValueError(
                            "Invalid type for ufunc output: should not be Quantity"
                        )
            kwargs['out'] = tuple(
                x.value if isinstance(x, Quantity) else x
                for x in out)

        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            if ufunc_units.wrap_output:
                return tuple(type(self)(x, result_dimension, result_scale) for x in result)
            else:
                return result
        elif method == 'at':
            return None
        else:
            if ufunc_units.wrap_output:
                return type(self)(result, result_dimension, result_scale)
            else:
                return result

    @_op
    def __add__(self, other: Any) -> "Quantity": ...

    @_op
    def __sub__(self, other: Any) -> "Quantity": ...

    @_op
    def __mul__(self, other: Any) -> "Quantity": ...

    @_op
    def __matmul__(self, other: Any) -> "Quantity": ...

    @_op
    def __truediv__(self, other: Any) -> "Quantity": ...

    @_op
    def __floordiv__(self, other: Any) -> "Quantity": ...

    @_op
    def __mod__(self, other: Any) -> "Quantity": ...

    @_op
    def __pow__(self, other: Any) -> "Quantity": ...

    @_op
    def __lshift__(self, other: Any) -> "Quantity": ...

    @_op
    def __rshift__(self, other: Any) -> "Quantity": ...

    @_op
    def __and__(self, other: Any) -> "Quantity": ...

    @_op
    def __xor__(self, other: Any) -> "Quantity": ...

    @_op
    def __or__(self, other: Any) -> "Quantity": ...

    @_op
    def __radd__(self, other: Any) -> "Quantity": ...

    @_op
    def __rsub__(self, other: Any) -> "Quantity": ...

    @_op
    def __rmul__(self, other: Any) -> "Quantity": ...

    @_op
    def __rmatmul__(self, other: Any) -> "Quantity": ...

    @_op
    def __rtruediv__(self, other: Any) -> "Quantity": ...

    @_op
    def __rfloordiv__(self, other: Any) -> "Quantity": ...

    @_op
    def __rmod__(self, other: Any) -> "Quantity": ...

    @_op
    def __rpow__(self, other: Any) -> "Quantity": ...

    @_op
    def __rlshift__(self, other: Any) -> "Quantity": ...

    @_op
    def __rrshift__(self, other: Any) -> "Quantity": ...

    @_op
    def __rand__(self, other: Any) -> "Quantity": ...

    @_op
    def __rxor__(self, other: Any) -> "Quantity": ...

    @_op
    def __ror__(self, other: Any) -> "Quantity": ...

    @_iop
    def __iadd__(self, other: Any) -> Any: ...

    @_iop
    def __isub__(self, other: Any) -> Any: ...

    @_iop
    def __imul__(self, other: Any) -> Any: ...

    @_iop
    def __imatmul__(self, other: Any) -> Any: ...

    @_iop
    def __itruediv__(self, other: Any) -> Any: ...

    @_iop
    def __ifloordiv__(self, other: Any) -> Any: ...

    @_iop
    def __imod__(self, other: Any) -> Any: ...

    @_iop
    def __ipow__(self, other: Any) -> Any: ...

    @_iop
    def __ilshift__(self, other: Any) -> Any: ...

    @_iop
    def __irshift__(self, other: Any) -> Any: ...

    @_iop
    def __iand__(self, other: Any) -> Any: ...

    @_iop
    def __ixor__(self, other: Any) -> Any: ...

    @_iop
    def __ior__(self, other: Any) -> Any: ...


_ufuncs[numpy.add] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.subtract] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.multiply] = _UfuncUnits(unit_map=_multiply_units)
_ufuncs[numpy.matmul] = _UfuncUnits(unit_map=_multiply_units)
_ufuncs[numpy.divide] = _UfuncUnits(unit_map=_divide_units)
_ufuncs[numpy.true_divide] = _UfuncUnits(unit_map=_divide_units)
_ufuncs[numpy.floor_divide] = _UfuncUnits(unit_map=_divide_units)
_ufuncs[numpy.negative] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.positive] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.power] = _UfuncUnits(unit_map=_power_units)
_ufuncs[numpy.float_power] = _UfuncUnits(unit_map=_power_units)
_ufuncs[numpy.remainder] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.mod] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.fmod] = _UfuncUnits(unit_map=_match_units)
# _ufuncs[numpy.divmod] = _UfuncUnits(unit_map=_match_units)  TODO: think about divmod
_ufuncs[numpy.absolute] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.fabs] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.rint] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.sign] = _UfuncUnits(unit_map=_match_units, wrap_output=False)
_ufuncs[numpy.conj] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.conjugate] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.sqrt] = _UfuncUnits(unit_map=lambda ufunc, x: _power_units(ufunc, x, Quantity(1 / 2, Scalar, x.scale)))
_ufuncs[numpy.square] = _UfuncUnits(unit_map=lambda ufunc, x: _power_units(ufunc, x, Quantity(2, Scalar, x.scale)))
_ufuncs[numpy.cbrt] = _UfuncUnits(unit_map=lambda ufunc, x: _power_units(ufunc, x, Quantity(1 / 3, Scalar, x.scale)))
_ufuncs[numpy.reciprocal] = _UfuncUnits(unit_map=lambda ufunc, x: _divide_units(ufunc, Quantity(1, Scalar, x.scale), x, offset=-1))
_ufuncs[numpy.greater] = _UfuncUnits(unit_map=_match_units, wrap_output=False)
_ufuncs[numpy.greater_equal] = _UfuncUnits(unit_map=_match_units, wrap_output=False)
_ufuncs[numpy.less] = _UfuncUnits(unit_map=_match_units, wrap_output=False)
_ufuncs[numpy.less_equal] = _UfuncUnits(unit_map=_match_units, wrap_output=False)
_ufuncs[numpy.not_equal] = _UfuncUnits(unit_map=_match_units, wrap_output=False)
_ufuncs[numpy.equal] = _UfuncUnits(unit_map=_match_units, wrap_output=False)
_ufuncs[numpy.maximum] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.minimum] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.fmax] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.fmin] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.isfinite] = _UfuncUnits(unit_map=_match_units, wrap_output=False)
_ufuncs[numpy.isinf] = _UfuncUnits(unit_map=_match_units, wrap_output=False)
_ufuncs[numpy.isnan] = _UfuncUnits(unit_map=_match_units, wrap_output=False)
_ufuncs[numpy.fabs] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.signbit] = _UfuncUnits(unit_map=_match_units, wrap_output=False)
_ufuncs[numpy.copysign] = _UfuncUnits(unit_map=lambda ufunc, x, y: _match_units(ufunc, x))
_ufuncs[numpy.nextafter] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.spacing] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.ldexp] = _UfuncUnits(unit_map=lambda ufunc, x, y: _scalar_units(ufunc, y, offset=1) and _match_units(ufunc, x))
