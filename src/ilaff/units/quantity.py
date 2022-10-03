import abc
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
import inspect
from itertools import chain
import numpy
from pathlib import Path
from typing import (
    cast, overload, Any, Optional, Tuple, Iterator, Iterable, Sequence, List,
    Type, Callable, Mapping, MutableMapping, Union, Sized, TextIO, BinaryIO, TypeVar,
)
import warnings
import pandas  # type: ignore
from pandas.util._decorators import cache_readonly  # type: ignore

from .dimension import Dimension, Scalar


class Scale(abc.ABC):
    @abc.abstractmethod
    def unit(self, dimension: Dimension) -> Tuple["Quantity", str]: ...


def _argument_name(index: int, labels: Mapping[int, str], offset: int, argument: str) -> str:
    position = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
    }
    if index in labels:
        return f"{labels[index]} {argument}"
    elif index + offset in position:
        return f"{position[index + 1 + offset]} {argument}"
    else:
        return f"{argument} {index + 1 + offset}"


def _scalar_units(func: Callable, *inputs: Union["Quantity", "QuantityIndex"], labels: Mapping[int, str] = {},
                  offset: int = 0, argument: str = "argument") -> Tuple[Dimension, Scale]:
    dimension = Scalar
    scale = inputs[0].scale
    for i, x in enumerate(inputs):
        if x.dimension != dimension:
            raise ValueError(
                f"Invalid mass dimension for {_argument_name(i, labels, offset, argument)}"
                f" of {func.__name__}: got {x.dimension}, expected {dimension}"
            )

    return dimension, scale


def _match_units(func: Callable, *inputs: Union["Quantity", "QuantityIndex"], labels: Mapping[int, str] = {},
                 offset: int = 0, argument: str = "argument") -> Tuple[Dimension, Scale]:
    for x in inputs:
        try:
            if numpy.any(x.value != 0):
                dimension = x.dimension
                scale = x.scale
                break
        except:
            dimension = x.dimension
            scale = x.scale
            break
    else:
        dimension = inputs[0].dimension
        scale = inputs[0].scale
    for i, x in enumerate(inputs):
        try:
            if not numpy.any(x.value != 0):
                continue
        except:
            pass
        if x.dimension != dimension:
            raise ValueError(
                f"Invalid mass dimension for {_argument_name(i, labels, offset, argument)}"
                f" of {func.__name__}: got {x.dimension}, expected {dimension}. \n {inputs}"
            )
        if x.dimension != Scalar and x.scale != scale:
            raise ValueError(
                f"Mismatched scale for {_argument_name(i, labels, offset, argument)} of {func.__name__}"
            )

    return dimension, scale


def _multiply_units(func: Callable, *inputs: Union["Quantity", "QuantityIndex"], labels: Mapping[int, str] = {},
                    offset: int = 0, argument: str = "argument") -> Tuple[Dimension, Scale]:
    dimension = inputs[0].dimension
    scale = inputs[0].scale
    for i, x in enumerate(inputs):
        if i > 0:
            if dimension == Scalar and x.dimension != Scalar:
                scale = x.scale
            dimension = dimension * x.dimension
            if x.dimension != Scalar and x.scale != scale:
                raise ValueError(
                    f"Mismatched scale for {_argument_name(i, labels, offset, argument)} of {func.__name__}"
                )

    return dimension, scale


def _divide_units(func: Callable, *inputs: Union["Quantity", "QuantityIndex"], labels: Mapping[int, str] = {},
                  offset: int = 0, argument: str = "argument") -> Tuple[Dimension, Scale]:
    dimension = inputs[0].dimension
    scale = inputs[0].scale
    for i, x in enumerate(inputs):
        if i > 0:
            if dimension == Scalar and x.dimension != Scalar:
                scale = x.scale
            dimension = dimension / x.dimension
            if x.dimension != Scalar and x.scale != scale:
                raise ValueError(
                    f"Mismatched scale for {_argument_name(i, labels, offset, argument)} of {func.__name__}"
                )

    return dimension, scale


def _power_units(func: Callable, *inputs: Union["Quantity", "QuantityIndex"], labels: Mapping[int, str] = {},
                 offset: int = 0, argument: str = "argument") -> Tuple[Dimension, Scale]:
    dimension = inputs[0].dimension
    scale = inputs[0].scale
    for i, x in enumerate(inputs):
        if i > 0:
            if x.dimension != Scalar:
                raise ValueError(
                    f"Invalid mass dimension for {_argument_name(i, labels, offset, argument)}"
                    f" of {func.__name__}: got {x.dimension}, expected {Scalar}"
                )
            dimension = dimension**x.value

    return dimension, scale


@dataclass(frozen=True)
class _UfuncUnits:
    unit_map: Callable[..., Tuple[Dimension, Scale]] = _scalar_units
    wrap_output: bool = True


@dataclass(frozen=True)
class _ArrayFuncUnits:
    unwrap_inputs: Tuple[str, ...] = ()
    unit_map: Callable[..., Tuple[Dimension, Scale]] = _scalar_units
    wrap_output: bool = True


_ufuncs: MutableMapping[numpy.ufunc, _UfuncUnits] = defaultdict(_UfuncUnits)
_arrayfuncs: MutableMapping[Callable, Callable] = {}

_upcast_types: List[Type] = []

# Pandas (Series)
try:
    from pandas import Series

    _upcast_types.append(Series)
except ImportError:
    pass

# xarray (DataArray, Dataset, Variable)
try:
    from xarray import DataArray, Dataset, Variable

    _upcast_types += [DataArray, Dataset, Variable]
except ImportError:
    DataArray = None  # type: ignore
    pass


T_Callable = TypeVar("T_Callable", bound=Callable)

def _check_implemented(fn: T_Callable) -> T_Callable:
    @wraps(fn)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        other = args[0]
        if type(other) in _upcast_types:
            return NotImplemented
        # pandas often gets to arrays of quantities [ Q_(1,"m"), Q_(2,"m")]
        # and expects Quantity * array[Quantity] should return NotImplemented
        elif isinstance(other, list) and other and isinstance(other[0], type(self)):
            return NotImplemented
        return fn(self, *args, **kwargs)

    return wrapped  # type: ignore


def _delegate_to(delegate: Any) -> Callable[[T_Callable], T_Callable]:
    def decorator(method: T_Callable) -> T_Callable:
        return getattr(delegate, method.__name__)  # type: ignore
    return decorator


def _delegate_as(fn: Callable) -> Callable[[T_Callable], T_Callable]:
    def decorator(method: T_Callable) -> T_Callable:
        return fn  # type: ignore
    return decorator


def _op(method: Callable[["Quantity", Any], Any]) -> Callable[["Quantity", Any], "Quantity"]:
    return _check_implemented(_delegate_to(numpy.lib.mixins.NDArrayOperatorsMixin)(method))


def _iop(method: Callable[["Quantity", Any], Any]) -> Callable[["Quantity", Any], Any]:
    wrapped = _check_implemented(_delegate_to(numpy.lib.mixins.NDArrayOperatorsMixin)(method))

    @wraps(method)
    def fn(self: "Quantity", other: Any) -> Any:
        if isinstance(self.value, numpy.ndarray):
            return wrapped(self, other)
        else:
            return NotImplemented
    return fn


ArrayLike = Any
Array = Any
Dtype = Any
Flags = Any


class UnitStrippedWarning(UserWarning):
    pass


@dataclass(frozen=True, eq=False, order=False)
class Quantity(numpy.lib.mixins.NDArrayOperatorsMixin, pandas.api.extensions.ExtensionArray):
    value: Any
    dimension: Dimension
    scale: Scale

    def __eq__(self, other: Any) -> Any:
        if type(other) in _upcast_types:
            return numpy.equal(self, other)

        if not isinstance(other, Quantity):
            other = Quantity(other, Scalar, self.scale)

        _ = _match_units(self.__eq__, self, other)

        return self.value == other.value

    def __hash__(self) -> Any:
        return hash(self.value)

    @staticmethod
    def from_string(string: str, **units: "Quantity") -> "Quantity":
        from .scale import fm, MeV, GeV, one
        _units = {
            'fm': fm,
            'MeV': MeV,
            'GeV': GeV,
        }
        _units.update(units)

        string = string.strip()

        unit = one
        for lbl, u in _units.items():
            if string[-len(lbl):] == lbl:
                unit = u
                string = string[:-len(lbl)].strip()
                break

        if string[0] == '[' and string[-1] == ']':
            return numpy.array(
                [float(s) for s in string[1:-1].replace(',', ' ').split()]
            ) * unit

        return float(string) * unit

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

    def in_unit(self, val: Union["Quantity", DataArray]) -> Any:
        try:
            if self.value == 0.0:
                return 0.0
        except:
            pass
        try:
            val = val.data
        except AttributeError:
            pass
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

    def __setitem__(self, key: Any, value: Any) -> None:
        if not isinstance(value, Quantity):
            value = Quantity(
                value,
                Scalar,
                self.scale,
            )
        _ = _match_units(self.__setitem__, self, value)
        self.value[key] = value.value

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

    def __array__(self, t: Optional[Dtype] = None) -> numpy.ndarray:
        warnings.warn(
            "The unit of the quantity is stripped when downcasting to ndarray.",
            UnitStrippedWarning,
            stacklevel=2,
        )
        return numpy.asarray(self.value)

    def __array_function__(self, func: Callable, types: Iterable[Type],
                           args: Iterable[Any], kwargs: Mapping[str, Any]) -> Any:
        if any((t in _upcast_types) for t in types):
            return NotImplemented
        func_units = _arrayfuncs.get(func)
        if func_units is None:
            return NotImplemented

        return func_units(args, kwargs)

    def __array_ufunc__(self, ufunc: numpy.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if any((type(arg) in _upcast_types) for arg in chain(inputs, kwargs.values())):
            return NotImplemented

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
                        raise TypeError(
                            f"Invalid type for '{ufunc.__name__}' output: should be Quantity"
                        )
                    if x.dimension != result_dimension:
                        raise ValueError(
                            f"Invalid mass dimension for '{ufunc.__name__}' output:"
                            f" was {x.dimension}, should be {result_dimension}"
                        )
                    if x.scale != result_scale:
                        raise ValueError(
                            f"Mismatched scales for '{ufunc.__name__}' output"
                        )
            else:
                for x in out:
                    if isinstance(x, Quantity):
                        raise TypeError(
                            f"Invalid type for '{ufunc.__name__}' output: should not be Quantity"
                        )
            kwargs['out'] = tuple(
                x.value if isinstance(x, Quantity) else x
                for x in out)

        return self._wrap_ufunc_output(
            getattr(ufunc, method)(*inputs, **kwargs),
            ufunc_units,
            result_dimension,
            result_scale,
            method,
        )

    def _wrap_ufunc_output(self, result: Any, ufunc_units: _UfuncUnits,
                           result_dimension: Dimension, result_scale: Scale,
                           method: str) -> Any:
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

    @property
    def shape(self) -> Tuple[int, ...]:
        try:
            return self.value.shape  # type: ignore
        except AttributeError:
            return ()

    @property
    def ndim(self) -> int:
        try:
            return self.value.ndim  # type: ignore
        except AttributeError:
            return 0

    @property
    def T(self) -> "Quantity":
        return Quantity(
            self.value.T,
            self.dimension,
            self.scale,
        )

    @property
    def dtype(self) -> Dtype:
        return self.value.dtype

    @property
    def flags(self) -> Flags:
        return self.value.flags

    @property
    def flat(self) -> Iterator["Quantity"]:
        for v in self.value.flat:
            yield Quantity(
                v,
                self.dimension,
                self.scale,
            )

    @property
    def imag(self) -> "Quantity":
        return Quantity(
            self.value.imag,
            self.dimension,
            self.scale,
        )

    @property
    def itemsize(self) -> int:
        return self.value.itemsize  # type: ignore

    @property
    def nbytes(self) -> int:
        return self.value.nbytes  # type: ignore

    @property
    def real(self) -> "Quantity":
        return Quantity(
            self.value.real,
            self.dimension,
            self.scale,
        )

    @property
    def size(self) -> int:
        return self.value.size  # type: ignore

    @property
    def strides(self) -> Tuple[int]:
        return self.value.strides  # type: ignore

    @_delegate_to(numpy)
    def any(self, axis: Optional[int] = None, out: Optional[numpy.ndarray] = None,
            keepdims: bool = False) -> Union[bool, numpy.ndarray]: ...

    @_delegate_to(numpy)
    def all(self, axis: Optional[int] = None, out: Optional[numpy.ndarray] = None,
            keepdims: bool = False) -> Union[bool, numpy.ndarray]: ...

    @_delegate_to(numpy)
    def argmax(self, axis: Optional[int] = None, out: Optional[numpy.ndarray] = None) -> numpy.ndarray: ...

    @_delegate_to(numpy)
    def argmin(self, axis: Optional[int] = None, out: Optional[numpy.ndarray] = None) -> numpy.ndarray: ...

    @_delegate_to(numpy)
    def argpartition(self, kth: Union[int, Sequence[int]], axis: Optional[int] = -1, kind: str = 'introselect',
                     order: Optional[Union[str, Sequence[str]]] = None) -> numpy.ndarray: ...

    @_delegate_to(numpy)
    def argsort(self, axis: int = -1, kind: Optional[str] = None,
                order: Optional[Union[str, Sequence[str]]] = None) -> numpy.ndarray: ...

    @_check_implemented
    def astype(self, dtype: Dtype, order: str = 'K', casting: str = 'unsafe',
               subok: bool = True, copy: bool = True) -> "Quantity":
        return Quantity(
            self.value.astype(dtype, casting=casting, subok=subok, copy=copy),
            self.dimension,
            self.scale,
        )

    @_check_implemented
    def byteswap(self, inplace: bool = False) -> "Quantity":
        return Quantity(
            self.byteswap(inplace),
            self.dimension,
            self.scale,
        )

    @_delegate_to(numpy)
    def clip(self, a_min: "Quantity", a_max: "Quantity",
             out: Optional["Quantity"] = None, **kwargs: Any) -> "Quantity": ...

    @_delegate_to(numpy)
    def compress(self, condition: Array, axis: Optional[int] = None,
                 out: Optional["Quantity"] = None) -> "Quantity": ...

    @_delegate_to(numpy)
    def conj(self) -> "Quantity": ...

    @_delegate_to(numpy)
    def conjugate(self) -> "Quantity": ...

    @_delegate_to(numpy)
    def copy(self, order: str = 'K') -> "Quantity": ...

    def __copy__(self) -> "QuantityIndex":
        return self.copy()

    def __deepcopy__(self, memo: Any = None) -> "QuantityIndex":
        return self.copy()

    @_delegate_to(numpy)
    def cumprod(self, axis: Optional[int] = None, dtype: Optional[Dtype] = None,
                out: Optional["Quantity"] = None) -> "Quantity": ...

    @_delegate_to(numpy)
    def cumsum(self, axis: Optional[int] = None, dtype: Optional[Dtype] = None,
               out: Optional["Quantity"] = None) -> "Quantity": ...

    @_delegate_to(numpy)
    def diagonal(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> "Quantity": ...

    @_delegate_to(numpy)
    def dot(self, b: "Quantity", out: Optional["Quantity"] = None) -> "Quantity": ...

    # TODO: implement pickling

    # def dump(self, file: Union[str, Path]) -> None:

    # def dumps(self) -> str:

    @_check_implemented
    def fill(self, value: "Quantity") -> None:
        _ = _match_units(self.fill, self, value, offset=-1)
        self.value.fill(value.value)

    @_check_implemented
    def flatten(self, order: str = 'C') -> "Quantity":
        return Quantity(
            self.value.flatten(order),
            self.dimension,
            self.scale,
        )

    @_check_implemented
    def item(self, *args: Union[int, Tuple[int, ...]]) -> "Quantity":
        return Quantity(
            self.value.item(*args),
            self.dimension,
            self.scale,
        )

    @overload
    def itemset(self, arg1: Any) -> None: ...

    @overload
    def itemset(self, arg1: Union[int, Tuple[int, ...]], arg2: Any) -> None: ...

    @_check_implemented
    def itemset(self, arg1: Any, arg2: Any = None) -> None:
        if arg2 is not None:
            val = arg2
            if not isinstance(val, Quantity):
                val = Quantity(val, Scalar, self.scale)
            _ = _match_units(self.itemset, self, arg2)
            self.value.itemset(arg1, val.value)
        else:
            val = arg1
            if not isinstance(val, Quantity):
                val = Quantity(val, Scalar, self.scale)
            _ = _match_units(self.itemset, self, arg1, offset=-1)
            self.value.itemset(arg1.value)

    @_delegate_as(numpy.amax)
    def max(self, axis: Optional[Union[int, Sequence[int]]] = None,
            out: Optional["Quantity"] = None, keepdims: Optional[bool] = None,
            initial: Optional["Quantity"] = None, where: Optional[ArrayLike] = None) -> "Quantity": ...

    @_delegate_to(numpy)
    def mean(self, axis: Optional[Union[int, Sequence[int]]] = None, dtype: Optional[Dtype] = None,
             out: Optional["Quantity"] = None, keepdims: Optional[bool] = None) -> "Quantity":
        return numpy.mean(self, axis, dtype, out, keepdims)  # type: ignore

    @_delegate_as(numpy.amin)
    def min(self, axis: Optional[Union[int, Sequence[int]]] = None,
            out: Optional["Quantity"] = None, keepdims: Optional[bool] = None,
            initial: Optional["Quantity"] = None, where: Optional[ArrayLike] = None) -> "Quantity": ...

    @_check_implemented
    def newbyteorder(self, new_order: str = 'S') -> "Quantity":
        return Quantity(
            self.value.newbyteorder(new_order),
            self.dimension,
            self.scale
        )

    def nonzero(self) -> numpy.ndarray:
        return self.value.nonzero()  # type: ignore

    @_delegate_to(numpy)
    def partition(self, kth: Union[int, Sequence[int]], axis: int = -1, kind: str = 'introselect',
                  order: Optional[Union[str, Sequence[str]]] = None) -> "Quantity": ...

    @_delegate_to(numpy)
    def prod(self, axis: Optional[Union[int, Sequence[int]]] = None, dtype: Optional[Dtype] = None,
             out: Optional["Quantity"] = None, keepdims: Optional[bool] = None,
             initial: Optional["Quantity"] = None, where: ArrayLike = True) -> "Quantity": ...

    @_delegate_to(numpy)
    def ptp(self, axis: Optional[Union[int, Sequence[int]]] = None,
            out: Optional["Quantity"] = None, keepdims: bool = False) -> "Quantity": ...

    @_delegate_to(numpy)
    def put(self, indices: ArrayLike, values: "Quantity", mode: str = 'raise') -> None: ...

    @_delegate_to(numpy)
    def ravel(self, order: str = 'C') -> "Quantity": ...

    @_delegate_to(numpy)
    def repeat(self, repeats: Union[int, Array], axis: Optional[int] = None) -> "Quantity": ...

    @_delegate_to(numpy)
    def reshape(self, shape: Union[int, Sequence[int]], order: str = 'C') -> "Quantity": ...

    @_check_implemented
    def resize(self, new_shape: Union[int, Sequence[int]], refcheck: bool = True) -> "Quantity":
        return Quantity(
            self.value.resize(new_shape, refcheck),
            self.dimension,
            self.scale,
        )

    @_delegate_to(numpy)
    def round(self, decimals: int = 0, out: Optional["Quantity"] = None) -> "Quantity": ...

    @_delegate_to(numpy)
    def searchsorted(self, v: "Quantity", side: str = 'left', sorter: Optional[ArrayLike] = None) -> "Quantity": ...

    @_check_implemented
    def setflags(self, write: Optional[bool] = None, align: Optional[bool] = None, uic: Optional[bool] = None) -> None:
        self.value.setflags(write, align, uic)

    @_check_implemented
    def sort(self, axis: int = -1, kind: Optional[str] = None, order: Optional[Union[str, Sequence[str]]] = None) -> None:
        self.value.sort(axis, kind, order)

    @_delegate_to(numpy)
    def squeeze(self, axis: Optional[int] = None) -> "Quantity": ...

    @_delegate_to(numpy)
    def std(self, axis: Optional[Union[int, Sequence[int]]] = None,
            dtype: Optional[Dtype] = None, out: Optional["Quantity"] = None,
            ddof: int = 0, keepdims: bool = False) -> "Quantity": ...

    @_delegate_to(numpy)
    def sum(self, axis: Optional[Union[int, Sequence[int]]] = None,
            dtype: Optional[Dtype] = None, out: Optional["Quantity"] = None,
            keepdims: Optional[bool] = None, initial: Optional["Quantity"] = None,
            where: ArrayLike = True) -> "Quantity": ...

    @_delegate_to(numpy)
    def swapaxes(self, axis1: int, axis2: int) -> "Quantity": ...

    @_delegate_to(numpy)
    def take(self, indices: ArrayLike, axis: Optional[int] = None,
             out: Optional["Quantity"] = None, mode: str = 'raise') -> "Quantity": ...

    def tolist(self) -> "Quantity":
        return Quantity(
            self.value.tolist(),
            self.dimension,
            self.scale,
        )

    @_delegate_to(numpy)
    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1,
              dtype: Optional[Dtype] = None, out: Optional["Quantity"] = None) -> "Quantity": ...

    def transpose(self, *axes: int) -> "Quantity":
        return Quantity(
            self.value.transpose(*axes),
            self.dimension,
            self.scale,
        )

    @_delegate_to(numpy)
    def var(self, axis: Optional[Union[int, Sequence[int]]] = None,
            dtype: Optional[Dtype] = None, out: Optional["Quantity"] = None,
            ddof: int = 0, keepdims: bool = False) -> "Quantity": ...

    @_check_implemented
    def view(self, *args: Any, **kwargs: Any) -> "Quantity":
        return Quantity(
            self.value.view(*args, **kwargs),
            self.dimension,
            self.scale,
        )

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

    @_delegate_as(numpy.isnan)
    def isna(self) -> numpy.ndarray: ...

    @classmethod
    def _from_sequence(cls, scalars: Sequence["Quantity"], dtype: Optional[Dtype] = None, copy: bool = False) -> "Quantity":
        dimension, scale = _match_units(cls._from_sequence, *scalars, argument="scalar")
        if copy:
            value = numpy.fromiter((q.value.copy() for q in scalars), dtype=dtype, count=len(scalars))  # type: ignore
        else:
            value = numpy.fromiter((q.value for q in scalars), dtype=dtype, count=len(scalars))  # type: ignore
        return cls(value, dimension, scale)

    @classmethod
    def _from_factorized(cls, values: numpy.ndarray, original: "Quantity") -> "Quantity":
        return cls(values, original.dimension, original.scale)

    def _values_for_factorize(self) -> Tuple[numpy.ndarray, Any]:
        return self.value, -1

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence["Quantity"]) -> "Quantity":
        return numpy.concatenate(to_concat)  # type: ignore

    def to_index(self) -> "QuantityIndex":
        return QuantityIndex(
            pandas.Index(self.value, self.dtype),
            self.dimension,
            self.scale,
            self.value.dtype,
        )


def in_unit(val: Any, unit: Union[Quantity, DataArray]) -> Any:
    try:
        result = val / unit
    except ValueError:
        raise ValueError("Can't convert units: incompatible scales")

    try:
        result_data = result.data
        result_dims = result.dims
    except AttributeError:
        result_data = result

    if result_data.dimension.mass_dim != 0:
        raise ValueError(
            "Can't convert units: incompatible mass dimensions {} and {}"
            .format(unit.dimension.mass_dim, val.dimension.mass_dim)
        )

    if result is result_data:
        return result.value

    return result.copy(data=result_data.value)


def _index_method(func: T_Callable) -> T_Callable:
    @wraps(func)
    def fn(self: "QuantityIndex", *args: Any, **kwargs: Any) -> "QuantityIndex":
        return QuantityIndex(
            getattr(self.index, func.__name__)(*args, **kwargs),
            self.dimension,
            self.scale,
            self.dtype,
        )
    return fn  # type: ignore


class QuantityIndex(Quantity, pandas.Index):
    def __new__(cls, index: pandas.Index, dimension: Dimension, scale: Scale, dtype: Dtype) -> Any:
        return Quantity.__new__(cls)

    def __init__(self, index: pandas.Index, dimension: Dimension, scale: Scale, dtype: Dtype) -> None:
        Quantity.__init__(self, index, dimension, scale)
        self._dtype = dtype

    def __repr__(self) -> str:
        class_name = type(self).__name__
        space = f"\n{' ' * (len(class_name) + 1)}"
        index = repr(self.value).replace("\n", space)
        args = f",{space}".join((index, repr(self.dimension), repr(self.scale)))
        return f"{class_name}({args})"

    @property
    def array(self) -> numpy.ndarray:
        return numpy.array(self.value.array, dtype=self._dtype)

    @property
    def dtype(self) -> Dtype:
        return self.value.dtype

    @property
    def values(self) -> Quantity:
        return Quantity(
                self.value.values,
            self.dimension,
            self.scale,
        )

    @_index_method
    def astype(self, dtype: Dtype, order: str = 'K', casting: str = 'unsafe',
               subok: bool = True, copy: bool = True) -> Quantity: ...

    # def take(self, indices: ArrayLike, axis: int = 0, allow_fill: bool = True, fill_value: Optional[Quantity] = None, **kwargs: Any) -> Quantity:
    #     if fill_value is not None:
    #         if isinstance(fill_value, Quantity):
    #             wrapped = fill_value
    #             unwrapped = fill_value.value
    #         else:
    #             wrapped = Quantity(fill_value, Scalar, self.scale)
    #             unwrapped = fill_value
    #         _ = _match_units(self.take, self, wrapped, labels={1: "fill_value"})
    #         fill_value = unwrapped
    #     return Quantity(
    #         self.index.take(indices, axis, allow_fill, fill_value, **kwargs),
    #         self.dimension,
    #         self.scale,
    #     )

    def set_names(self, names: Union[str, Sequence[str]], *, level: Optional[Union[int, str, Sequence[Union[int, str]]]] = None, inplace: bool = False) -> "QuantityIndex":
        if inplace:
            self.value.set_names(names, level=level, inplace=True)
            return self
        else:
            return QuantityIndex(
                self.value.set_names(names, level=level, inplace=False),
                self.dimension,
                self.scale,
                self._dtype,
            )

    def __getitem__(self, key: Any) -> Union[Quantity, "QuantityIndex"]:
        value = self.value[key]
        if isinstance(value, pandas.Index):
            return QuantityIndex(
                value,
                self.dimension,
                self.scale,
                self._dtype,
            )
        else:
            return Quantity(
                value,
                self.dimension,
                self.scale,
            )

    def copy(self, name: Optional[str] = None, deep: bool = False, dtype: Optional[Dtype] = None, names: Optional[Sequence[str]] = None) -> "QuantityIndex":
        return QuantityIndex(
            self.value.copy(name, deep, dtype, names),
            self.dimension,
            self.scale,
            self._dtype,
        )

    def __copy__(self) -> "QuantityIndex":
        return self.copy(deep=False)

    def __deepcopy__(self, memo: Any = None) -> "QuantityIndex":
        return self.copy(deep=True)

    def equals(self, other: pandas.api.extensions.ExtensionArray) -> bool:
        if not isinstance(other, QuantityIndex):
            other = QuantityIndex(other, Scalar, self.scale, self._dtype)
        return self.dimension == other.dimension and self.scale == other.scale and self.value.equals(other.value)  # type: ignore


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
_ufuncs[numpy.sqrt] = _UfuncUnits(unit_map=lambda f, x: _power_units(f, x, Quantity(1 / 2, Scalar, x.scale)))
_ufuncs[numpy.square] = _UfuncUnits(unit_map=lambda f, x: _power_units(f, x, Quantity(2, Scalar, x.scale)))
_ufuncs[numpy.cbrt] = _UfuncUnits(unit_map=lambda f, x: _power_units(f, x, Quantity(1 / 3, Scalar, x.scale)))
_ufuncs[numpy.reciprocal] = _UfuncUnits(unit_map=lambda f, x: _divide_units(f, Quantity(1, Scalar, x.scale), x, offset=-1))
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
_ufuncs[numpy.copysign] = _UfuncUnits(unit_map=lambda f, x, y: _match_units(f, x))
_ufuncs[numpy.nextafter] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.spacing] = _UfuncUnits(unit_map=_match_units)
_ufuncs[numpy.ldexp] = _UfuncUnits(unit_map=lambda f, x, y: _scalar_units(f, y, offset=1) and _match_units(f, x))


# TODO: understand why type: ignore is required here
def _unwrap_annotation(parameter: inspect.Parameter) -> Type:
    if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
        return Sequence[parameter.annotation]  # type: ignore
    elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
        return Mapping[str, parameter.annotation]  # type: ignore
    else:
        return parameter.annotation  # type: ignore


# TODO: consider performance
def _array_func(func: Callable) -> Callable[[Callable], Callable]:
    def decorator(unit_map: Callable) -> Callable:
        signature = inspect.signature(unit_map)
        annotation: Mapping[str, Type] = {
            name: _unwrap_annotation(param)
            for name, param in signature.parameters.items()
        }

        def wrapper(args: Iterable[Any], kwargs: Mapping[str, Any]) -> Any:
            bound = signature.bind(*args, **kwargs)
            try:
                scale = next(iter(
                    argument
                    for arguments in (
                        [arg]
                        if annotation[name] == Quantity else
                        arg
                        if annotation[name] == Sequence[Quantity] else
                        arg.values()
                        if annotation[name] == Mapping[str, Quantity] else
                        ([] if arg is None else [arg])
                        if annotation[name] == Optional[Quantity] else
                        []
                        for name, arg in bound.arguments.items()
                    )
                    for argument in arguments
                    if argument is not None and isinstance(argument, Quantity)
                )).scale
            except StopIteration:
                raise TypeError(f"Unexpected Quantity passed to {func.__name__}")
            wrapped_arguments: MutableMapping[str, Any] = {}
            unwrapped_arguments: MutableMapping[str, Any] = {}
            for name, arg in bound.arguments.items():
                ann = annotation[name]
                if ann == Quantity:
                    if isinstance(arg, Quantity):
                        wrapped_arguments[name] = arg
                        unwrapped_arguments[name] = arg.value
                    else:
                        wrapped_arguments[name] = Quantity(arg, Scalar, scale)
                        unwrapped_arguments[name] = arg
                elif ann == Optional[Quantity]:
                    if arg is None:
                        wrapped_arguments[name] = arg
                        unwrapped_arguments[name] = arg
                    elif isinstance(arg, Quantity):
                        wrapped_arguments[name] = arg
                        unwrapped_arguments[name] = arg.value
                    else:
                        wrapped_arguments[name] = Quantity(arg, Scalar, scale)
                        unwrapped_arguments[name] = arg
                elif ann == Sequence[Quantity]:
                    wrapped_arguments[name], unwrapped_arguments[name] = zip(*(
                        (a, a.value)
                        if isinstance(a, Quantity) else
                        (Quantity(a, Scalar, scale), a)
                        for a in arg
                    ))
                elif ann == Mapping[str, Quantity]:
                    wrapped_arguments[name], unwrapped_arguments[name] = map(dict, zip(*(
                        ((key, a), (key, a.value))
                        if isinstance(a, Quantity) else
                        ((key, Quantity(a, Scalar, scale)), (key, a))
                        for key, a in arg.items()
                    )))
                else:
                    if isinstance(arg, Quantity):
                        raise TypeError(f"Unexpected Quantity passed to argument '{name}' of {func.__name__}")
                    wrapped_arguments[name] = arg
                    unwrapped_arguments[name] = arg
            wrapped = inspect.BoundArguments(signature, wrapped_arguments)  # type:ignore
            unwrapped = inspect.BoundArguments(signature, unwrapped_arguments)  # type:ignore
            output = unit_map(*wrapped.args, **wrapped.kwargs)
            result = func(*unwrapped.args, **unwrapped.kwargs)
            if output is None:
                return result
            if type(output) == tuple:
                return Quantity(result, *output)
            if type(result) == tuple:
                if len(output) == len(result):
                    return tuple(r if o is None else Quantity(r, *o) for r, o in zip(result, output))
            elif type(result) == list:
                if len(output) == len(result):
                    return [r if o is None else Quantity(r, *o) for r, o in zip(result, output)]
            else:
                if len(output) == 1:
                    return result if output[0] is None else Quantity(result, *output[0])

            raise RuntimeError(f"Return value of {func.__name__} does not match expected format")
        _arrayfuncs[func] = wrapper
        return func

    return decorator


def _einsum_func(func: Callable) -> Callable[[Callable], Callable]:
    def decorator(unit_map: Callable) -> Callable:
        def wrapper(args: Iterable[Any], kwargs: Mapping[str, Any]) -> Any:
            operands = args
            try:
                scale = next(iter(
                    argument
                    for argument in operands
                    if argument is not None and isinstance(argument, Quantity)
                )).scale
            except StopIteration:
                print(operands)
                raise TypeError(f"Unexpected Quantity passed to {func.__name__}")

            unwrapped = [
                argument.value
                if isinstance(argument, Quantity) else
                argument
                for argument in operands
            ]

            wrapped = [
                Quantity(argument, Scalar, scale)
                for argument in operands
                if not isinstance(argument, (str, list))
            ]
            labels = [
                {
                    1: "first",
                    2: "second",
                    3: "third",
                    4: "fourth",
                    5: "fifth",
                    6: "sixth",
                }.get(i + 1, f"{i+1}th")
                for i, argument in enumerate(operands)
                if not isinstance(argument, (str, list))
            ]

            output = unit_map(wrapped, dict(enumerate(labels)))
            result = func(*unwrapped, **kwargs)
            out = kwargs.get('out')
            if out is not None:
                if not isinstance(out, Quantity):
                    out = Quantity(out, Scalar, scale)
                _ = _match_units(func, Quantity(0, *output), out, labels={1: "'out'"})
            if output is None:
                return result
            if type(output) is tuple:
                return Quantity(result, *output)

            raise RuntimeError(f"Return value of {func.__name__} does not match expected format")
        _arrayfuncs[func] = wrapper
        return func

    return decorator


@_array_func(numpy.take)
def take(a: Quantity, indices: ArrayLike, axis: Optional[int] = None,
         out: Optional[Quantity] = None, mode: str = 'raise') -> Tuple[Dimension, Scale]:
    if out is not None:
        return _match_units(numpy.take, a, out, labels={1: "'out'"})
    else:
        return a.dimension, a.scale


@_array_func(numpy.reshape)
def reshape(a: Quantity, newshape: Union[int, Sequence[int]], order: str = 'C') -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.choose)
def choose(a: Array, choices: Sequence[Quantity], out: Optional[Quantity] = None,
           mode: str = 'raise') -> Tuple[Dimension, Scale]:
    if out is not None:
        _ = _match_units(numpy.choose, choices[0], out, labels={1: "'out'"})
    return _match_units(numpy.choose, *choices, argument="choice")


@_array_func(numpy.repeat)
def repeat(a: Quantity, repeats: Union[int, Array], axis: Optional[int] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.put)
def put(a: Quantity, ind: ArrayLike, v: Quantity, mode: str = 'raise') -> None:
    _ = _match_units(numpy.put, a, v, offset=1)
    return None


@_array_func(numpy.swapaxes)
def swapaxes(a: Quantity, axis1: int, axis2: int) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.transpose)
def transpose(a: Quantity, axes: Optional[Sequence[int]] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.partition)
def partition(a: Quantity, kth: Union[int, Sequence[int]], axis: int = -1, kind: str = 'introselect',
              order: Optional[Union[str, Sequence[str]]] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.argpartition)
def argpartition(a: Quantity, kth: Union[int, Sequence[int]], axis: int = -1, kind: str = 'introselect',
                 order: Optional[Union[str, Sequence[str]]] = None) -> None:
    return None


@_array_func(numpy.sort)
def sort(a: Quantity, axis: Optional[int] = -1, kind: Optional[str] = None,
         order: Optional[Union[str, Sequence[str]]] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.argsort)
def argsort(a: Quantity, axis: Optional[int] = -1, kind: Optional[str] = None,
            order: Optional[Union[str, Sequence[str]]] = None) -> None:
    return None


@_array_func(numpy.argmax)
def argmax(a: Quantity, axis: Optional[int] = None, out: Optional[Array] = None) -> None:
    return None


@_array_func(numpy.argmin)
def argmin(a: Quantity, axis: Optional[int] = None, out: Optional[Array] = None) -> None:
    return None


@_array_func(numpy.searchsorted)
def searchsorted(a: Quantity, v: Quantity, side: str = 'left',
                 sorter: Optional[ArrayLike] = None) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.searchsorted, a, v)


@_array_func(numpy.resize)
def resize(a: Quantity, new_shape: Union[int, Sequence[int]]) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.squeeze)
def squeeze(a: Quantity, axis: Optional[int] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.diagonal)
def diagonal(a: Quantity, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.trace)
def trace(a: Quantity, offset: int = 0, axis1: int = 0, axis2: int = 1,
          dtype: Optional[Dtype] = None, out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.trace, a, out, labels={1: "'out'"})


@_array_func(numpy.ravel)
def ravel(a: Quantity, order: str = 'C') -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.nonzero)
def nonzero(a: Quantity) -> None:
    return None


@_array_func(numpy.shape)
def shape(a: Quantity) -> None:
    return None


@_array_func(numpy.compress)
def compress(condition: Array, a: Quantity, axis: Optional[int] = None,
             out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.compress, a, out, labels={1: "'out'"}, offset=1)


@_array_func(numpy.clip)
def clip(a: Quantity, a_min: Quantity, a_max: Quantity,
         out: Optional[Quantity] = None, **kwargs: Any) -> Tuple[Dimension, Scale]:
    if out is None:
        return _match_units(numpy.clip, a, a_min, a_max)
    else:
        return _match_units(numpy.clip, a, a_min, a_max, out, labels={3: "'out'"})


@_array_func(numpy.sum)
def sum(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None, dtype: Optional[Dtype] = None,
        out: Optional[Quantity] = None, keepdims: Optional[bool] = None,
        initial: Optional[Quantity] = None, where: Optional[ArrayLike] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        if initial is None:
            return a.dimension, a.scale
        else:
            return _match_units(numpy.sum, a, initial, labels={1: "'initial'"})
    else:
        if initial is None:
            return _match_units(numpy.sum, a, out, labels={1: "'out'"})
        else:
            return _match_units(numpy.sum, a, out, initial, labels={1: "'out'", 2: "'initial'"})


@_array_func(numpy.any)
def _any(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
         out: Optional[ArrayLike] = None, keepdims: Optional[bool] = None) -> None:
    return None


@_array_func(numpy.all)
def _all(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
         out: Optional[ArrayLike] = None, keepdims: Optional[bool] = None) -> None:
    return None


@_array_func(numpy.cumsum)
def cumsum(a: Quantity, axis: Optional[int] = None, dtype: Optional[Dtype] = None,
           out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.cumsum, a, out, labels={1: "'out'"})


@_array_func(numpy.ptp)
def ptp(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
        out: Optional[Quantity] = None, keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.ptp, a, out, labels={1: "'out'"})


@_array_func(numpy.amax)
def amax(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
         out: Optional[Quantity] = None, keepdims: Optional[bool] = None,
         initial: Optional[Quantity] = None, where: Optional[ArrayLike] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        if initial is None:
            return a.dimension, a.scale
        else:
            return _match_units(numpy.amax, a, initial, labels={1: "'initial'"})
    else:
        if initial is None:
            return _match_units(numpy.amax, a, out, labels={1: "'out'"})
        else:
            return _match_units(numpy.amax, a, out, initial, labels={1: "'out'", 2: "'initial'"})


@_array_func(numpy.amin)
def amin(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
         out: Optional[Quantity] = None, keepdims: Optional[bool] = None,
         initial: Optional[Quantity] = None, where: Optional[ArrayLike] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        if initial is None:
            return a.dimension, a.scale
        else:
            return _match_units(numpy.amin, a, initial, labels={1: "'initial'"})
    else:
        if initial is None:
            return _match_units(numpy.amin, a, out, labels={1: "'out'"})
        else:
            return _match_units(numpy.amin, a, out, initial, labels={1: "'out'", 2: "'initial'"})


@_array_func(numpy.alen)  # type: ignore
def alen(a: Quantity) -> None:
    return None


@_array_func(numpy.prod)
def prod(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None, dtype: Optional[Dtype] = None,
         out: Optional[Quantity] = None, keepdims: Optional[bool] = None,
         initial: Optional[Quantity] = None, where: Optional[ArrayLike] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        if initial is None:
            return a.dimension, a.scale
        else:
            return _match_units(numpy.prod, a, initial, labels={1: "'initial'"})
    else:
        if initial is None:
            return _match_units(numpy.prod, a, out, labels={1: "'out'"})
        else:
            return _match_units(numpy.prod, a, out, initial, labels={1: "'out'", 2: "'initial'"})


@_array_func(numpy.cumprod)
def cumprod(a: Quantity, axis: Optional[int] = None, dtype: Optional[Dtype] = None,
            out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.cumprod, a, out, labels={1: "'out'"})


@_array_func(numpy.ndim)
def ndim(a: Quantity) -> None:
    return None


@_array_func(numpy.size)
def size(a: Quantity, axis: Optional[int] = None) -> None:
    return None


@_array_func(numpy.around)
def around(a: Quantity, decimals: int = 0, out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.around, a, out, labels={1: "'out'"})


@_array_func(numpy.mean)
def mean(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
         dtype: Optional[Dtype] = None, out: Optional[Quantity] = None,
         keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.mean, a, out, labels={1: "'out'"})


@_array_func(numpy.std)
def std(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
        dtype: Optional[Dtype] = None, out: Optional[Quantity] = None,
        ddof: int = 0, keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.std, a, out, labels={1: "'out'"})


@_array_func(numpy.var)
def var(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
        dtype: Optional[Dtype] = None, out: Optional[Quantity] = None,
        ddof: int = 0, keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension**2, a.scale
    else:
        dimension, scale = _match_units(numpy.var, a, out, labels={1: "'out'"})
        return dimension**2, scale


@_array_func(numpy.round)
def round(a: Quantity, decimals: int = 0, out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.round, a, out, labels={1: "'out'"})


@_array_func(numpy.product)
def product(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
            dtype: Optional[Dtype] = None, out: Optional[Quantity] = None,
            keepdims: Optional[bool] = None, initial: Optional[Quantity] = None,
            where: Optional[ArrayLike] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        if initial is None:
            return a.dimension, a.scale
        else:
            return _match_units(numpy.product, a, initial, labels={1: "'initial'"})
    else:
        if initial is None:
            return _match_units(numpy.product, a, out, labels={1: "'out'"})
        else:
            return _match_units(numpy.product, a, out, initial, labels={1: "'out'", 2: "'initial'"})


@_array_func(numpy.cumproduct)
def cumproduct(a: Quantity, axis: Optional[int] = None, dtype: Optional[Dtype] = None,
               out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.cumproduct, a, out, labels={1: "'out'"})


@_array_func(numpy.sometrue)  # type: ignore
def sometrue(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
             out: Optional[ArrayLike] = None, keepdims: Optional[bool] = None) -> None:
    return None


@_array_func(numpy.alltrue)  # type: ignore
def alltrue(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
            out: Optional[ArrayLike] = None, keepdims: Optional[bool] = None) -> None:
    return None


@_array_func(numpy.linspace)
def linspace(start: Quantity, stop: Quantity, num: int = 50,
             endpoint: bool = True, retstep: bool = False, dtype: Optional[Dtype] = None,
             axis: int = 0) -> Union[Tuple[Dimension, Scale], List[Tuple[Dimension, Scale]]]:
    dimension, scale = _match_units(numpy.linspace, start, stop)
    if retstep:
        return [(dimension, scale), (dimension, scale)]
    return dimension, scale


@_array_func(numpy.logspace)
def logspace(start: Quantity, stop: Quantity, num: int = 50,
             endpoint: bool = True, retstep: bool = False, dtype: Optional[Dtype] = None,
             axis: int = 0) -> Union[Tuple[Dimension, Scale], List[Tuple[Dimension, Scale]]]:
    dimension, scale = _match_units(numpy.logspace, start, stop)
    if retstep:
        return [(dimension, scale), (dimension, scale)]
    return dimension, scale


@_array_func(numpy.geomspace)
def geomspace(start: Quantity, stop: Quantity, num: int = 50,
              endpoint: bool = True, retstep: bool = False, dtype: Optional[Dtype] = None,
              axis: int = 0) -> Union[Tuple[Dimension, Scale], List[Tuple[Dimension, Scale]]]:
    dimension, scale = _match_units(numpy.geomspace, start, stop)
    if retstep:
        return [(dimension, scale), (dimension, scale)]
    return dimension, scale


@_array_func(numpy.empty_like)
def empty_like(prototype: Quantity, dtype: Optional[Dtype] = None, order: str = 'K', subok: bool = True,
               shape: Optional[Union[int, Sequence[int]]] = None) -> Tuple[Dimension, Scale]:
    return prototype.dimension, prototype.scale


@_array_func(numpy.concatenate)
def concatenate(arrays: Sequence[Quantity], axis: int = 0, out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is not None:
        _ = _match_units(numpy.concatenate, arrays[0], out, labels={1: "'out'"})
    return _match_units(numpy.concatenate, *arrays, argument="array")


@_array_func(numpy.inner)
def inner(a: Quantity, b: Quantity) -> Tuple[Dimension, Scale]:
    return _multiply_units(numpy.inner, a, b)


@_array_func(numpy.where)
def where(condition: ArrayLike, x: Optional[Quantity] = None, y: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if x is None:
        if y is None:
            raise TypeError(f"Unexpected Quantity passed to {numpy.inner.__name__}")
        else:
            return _match_units(numpy.where, y, labels={0: "'y'"})
    else:
        if y is None:
            return _match_units(numpy.where, x, labels={0: "'x'"})
        else:
            return _match_units(numpy.where, x, y, labels={0: "'x'", 1: "'y'"})


@_array_func(numpy.lexsort)
def lexsort(keys: Quantity, axis: int = -1) -> None:
    return None


@_array_func(numpy.min_scalar_type)
def min_scalar_type(a: Quantity) -> None:
    return None


@_array_func(numpy.result_type)
def result_type(*args: Quantity) -> None:
    return None


@_array_func(numpy.dot)
def dot(a: Quantity, b: Quantity, out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    dimension, scale = _multiply_units(numpy.dot, a, b)
    if out is not None:
        if out.dimension != dimension:
            raise ValueError(
                f"Invalid mass dimension for argument 'out' of {numpy.dot.__name__}:"
                f" got {out.dimension}, expected {dimension}"
            )
        if out.dimension != Scalar and out.scale != scale:
            raise ValueError(
                f"Mismatched scale for argument 'out' of {numpy.dot.__name__}"
            )
    return dimension, scale


@_array_func(numpy.vdot)
def vdot(a: Quantity, b: Quantity) -> Tuple[Dimension, Scale]:
    return _multiply_units(numpy.vdot, a, b)


@_array_func(numpy.copyto)
def copyto(dst: Quantity, src: Quantity, casting: str = 'same_kind', where: ArrayLike = True) -> None:
    _ = _match_units(numpy.copyto, dst, src)
    return None


@_array_func(numpy.putmask)
def putmask(a: Quantity, mask: ArrayLike, values: Quantity) -> None:
    _ = _match_units(numpy.putmask, a, values)
    return None


@_array_func(numpy.shares_memory)
def shares_memory(a: Quantity, b: Quantity, max_work: Optional[int] = None) -> None:
    _ = _match_units(numpy.shares_memory, a, b)
    return None


@_array_func(numpy.may_share_memory)
def may_share_memory(a: Quantity, b: Quantity, max_work: Optional[int] = None) -> None:
    _ = _match_units(numpy.may_share_memory, a, b)
    return None


@_array_func(numpy.zeros_like)
def zeros_like(a: Quantity, dtype: Optional[Dtype] = None, order: str = 'K', subok: bool = True,
               shape: Optional[Union[int, Sequence[int]]] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.ones_like)
def ones_like(a: Quantity, dtype: Optional[Dtype] = None, order: str = 'K', subok: bool = True,
              shape: Optional[Union[int, Sequence[int]]] = None) -> Tuple[Dimension, Scale]:
    return Scalar, a.scale


@_array_func(numpy.full_like)
def full_like(a: Quantity, fill_value: Quantity, dtype: Optional[Dtype] = None, order: str = 'K',
              subok: bool = True, shape: Optional[Union[int, Sequence[int]]] = None) -> Tuple[Dimension, Scale]:
    return fill_value.dimension, fill_value.scale


@_array_func(numpy.count_nonzero)
def count_nonzero(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None, *, keepdims: bool = False) -> None:
    return None


@_array_func(numpy.argwhere)
def argwhere(a: Quantity) -> None:
    return None


@_array_func(numpy.flatnonzero)
def flatnonzero(a: Quantity) -> None:
    return None


@_array_func(numpy.correlate)
def correlate(a: Quantity, v: Quantity, mode: str = 'valid') -> Tuple[Dimension, Scale]:
    return _multiply_units(numpy.correlate, a, v)


@_array_func(numpy.convolve)
def convolve(a: Quantity, v: Quantity, mode: str = 'full') -> Tuple[Dimension, Scale]:
    return _multiply_units(numpy.correlate, a, v)


@_array_func(numpy.outer)
def outer(a: Quantity, b: Quantity, out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    dimension, scale = _multiply_units(numpy.outer, a, b)
    if out is not None:
        if out.dimension != dimension:
            raise ValueError(
                f"Invalid mass dimension for argument 'out' of {numpy.outer.__name__}:"
                f" got {out.dimension}, expected {dimension}"
            )
        if out.dimension != Scalar and out.scale != scale:
            raise ValueError(
                f"Mismatched scale for argument 'out' of {numpy.outer.__name__}"
            )
    return dimension, scale


@_array_func(numpy.tensordot)
def tensordot(a: Quantity, b: Quantity, axes: Union[int, ArrayLike] = 2) -> Tuple[Dimension, Scale]:
    return _multiply_units(numpy.tensordot, a, b)


@_array_func(numpy.roll)
def roll(a: Quantity, shift: Union[int, Sequence[int]],
         axis: Optional[Union[int, Sequence[int]]] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.rollaxis)
def rollaxis(a: Quantity, axis: int, start: int = 0) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.moveaxis)
def moveaxis(a: Quantity, source: Union[int, Sequence[int]],
             destination: Union[int, Sequence[int]]) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.cross)
def cross(a: Quantity, b: Quantity, axisa: int = -1, axisb: int = -1,
          axisx: int = -1, axis: Optional[int] = None) -> Tuple[Dimension, Scale]:
    return _multiply_units(numpy.cross, a, b)


@_array_func(numpy.allclose)
def allclose(a: Quantity, b: Quantity, rtol: Optional[Quantity] = None,
             atol: Optional[Quantity] = None, equal_nan: bool = False) -> None:
    if rtol is not None:
        _ = _scalar_units(numpy.allclose, rtol, labels={0: "'rtol'"})
    if atol is not None:
        _ = _match_units(numpy.allclose, a, b, atol, labels={2: "'atol'"})
    else:
        _ = _match_units(numpy.allclose, a, b)
    return None


@_array_func(numpy.isclose)
def isclose(a: Quantity, b: Quantity, rtol: Optional[Quantity] = None,
            atol: Optional[Quantity] = None, equal_nan: bool = False) -> None:
    if rtol is not None:
        _ = _scalar_units(numpy.isclose, rtol, labels={0: "'rtol'"})
    if atol is not None:
        _ = _match_units(numpy.isclose, a, b, atol, labels={2: "'atol'"})
    else:
        _ = _match_units(numpy.isclose, a, b)
    return None


@_array_func(numpy.array_equal)
def array_equal(a1: Quantity, a2: Quantity, equal_nan: bool = False) -> None:
    _ = _match_units(numpy.array_equal, a1, a2)
    return None


@_array_func(numpy.array_equiv)
def array_equiv(a1: Quantity, a2: Quantity) -> None:
    _ = _match_units(numpy.array_equiv, a1, a2)
    return None


@_array_func(numpy.atleast_1d)
def atleast_1d(*arys: Quantity) -> List[Tuple[Dimension, Scale]]:
    return [(a.dimension, a.scale) for a in arys]


@_array_func(numpy.atleast_2d)
def atleast_2d(*arys: Quantity) -> List[Tuple[Dimension, Scale]]:
    return [(a.dimension, a.scale) for a in arys]


@_array_func(numpy.atleast_3d)
def atleast_3d(*arys: Quantity) -> List[Tuple[Dimension, Scale]]:
    return [(a.dimension, a.scale) for a in arys]


@_array_func(numpy.vstack)
def vstack(tup: Sequence[Quantity]) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.vstack, *tup)


@_array_func(numpy.hstack)
def hstack(tup: Sequence[Quantity]) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.hstack, *tup)


@_array_func(numpy.stack)
def stack(arrays: Sequence[Quantity], axis: int = 0, out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is not None:
        return _match_units(numpy.stack, *arrays, out, labels={len(arrays): "'out'"})
    else:
        return _match_units(numpy.stack, *arrays)


@_array_func(numpy.rot90)
def rot90(m: Quantity, k: int = 1, axes: ArrayLike = (0, 1)) -> Tuple[Dimension, Scale]:
    return m.dimension, m.scale


@_array_func(numpy.flip)
def flip(m: Quantity, axis: Optional[Union[int, Sequence[int]]] = None) -> Tuple[Dimension, Scale]:
    return m.dimension, m.scale


@_array_func(numpy.average)
def average(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
            weights: Optional[Quantity] = None, returned: bool = False) -> List[Tuple[Dimension, Scale]]:
    if returned:
        if weights is None:
            return [
                (a.dimension, a.scale),
                (Scalar, a.scale),
            ]

        return [
            (a.dimension, a.scale),
            (weights.dimension, weights.scale),
        ]
    return [(a.dimension, a.scale)]


# TODO: Consider performance
@_array_func(numpy.piecewise)
def piecewise(x: Quantity, condlist: Sequence[Union[bool, Array]],
              funclist: Sequence[Callable], *args: Any, **kw: Any) -> Tuple[Dimension, Scale]:
    mapped = (
        func(x[nonzero[0]], *args, **kw)
        for nonzero, func in zip(
            (numpy.nonzero(cond) for cond in condlist),
            funclist,
        )
        if len(nonzero) > 0
    )
    return _match_units(numpy.piecewise, *mapped, argument="function")


@_array_func(numpy.select)
def select(condlist: Sequence[Array], choicelist: Sequence[Quantity],
           default: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if default is not None:
        _ = _match_units(numpy.select, choicelist[0], default, labels={1: "'default'"})
    return _match_units(numpy.select, *choicelist, argument="choice")


@_array_func(numpy.copy)
def copy(a: Quantity, order: str = 'K', subok: bool = False) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.gradient)
def gradient(f: Quantity, *varargs: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
             edge_order: int = 1) -> List[Tuple[Dimension, Scale]]:
    shape = numpy.shape(f.value)
    N = len(shape)
    if len(varargs) == 0:
        return [(f.dimension, f.scale)] * N
    elif len(varargs) == 1:
        return [_divide_units(numpy.gradient, f, varargs[0])] * N
    else:
        return [_divide_units(numpy.gradient, f, arg, offset=i) for i, arg in enumerate(varargs)]


@_array_func(numpy.diff)
def diff(a: Quantity, n: int = 1, axis: int = -1, prepend: Optional[Quantity] = None,
         append: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if prepend is None:
        if append is None:
            return a.dimension, a.scale
        else:
            return _match_units(numpy.diff, a, append, labels={1: "'append'"})
    else:
        if append is None:
            return _match_units(numpy.diff, a, prepend, labels={1: "'prepend'"})
        else:
            return _match_units(numpy.diff, a, prepend, append, labels={1: "'prepend'", 2: "'append'"})


@_array_func(numpy.interp)
def interp(x: Quantity, xp: Quantity, fp: Quantity, left: Optional[Quantity] = None,
           right: Optional[Quantity] = None, period: Optional[float] = None) -> Tuple[Dimension, Scale]:
    _ = _match_units(numpy.interp, x, xp)
    if left is None:
        if right is None:
            return fp.dimension, fp.scale
        else:
            return _match_units(numpy.interp, fp, right, labels={1: "'right'"})
    else:
        if right is None:
            return _match_units(numpy.interp, fp, left, labels={1: "'left'"})
        else:
            return _match_units(numpy.interp, fp, left, right, labels={1: "'left'", 2: "'right'"})


@_array_func(numpy.angle)
def angle(z: Quantity, deg: bool = False) -> None:
    return None


@_array_func(numpy.unwrap)
def unwrap(p: Quantity, discont: float = numpy.pi, axis: int = -1) -> Tuple[Dimension, Scale]:
    return _scalar_units(numpy.unwrap, p)


@_array_func(numpy.sort_complex)
def sort_complex(a: Quantity) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.trim_zeros)
def trim_zeros(filt: Quantity, trim: str = 'fb') -> Tuple[Dimension, Scale]:
    return filt.dimension, filt.scale


@_array_func(numpy.extract)
def extract(condition: ArrayLike, arr: Quantity) -> Tuple[Dimension, Scale]:
    return arr.dimension, arr.scale


@_array_func(numpy.place)
def place(arr: Quantity, mask: ArrayLike, vals: Quantity) -> None:
    _ = _match_units(numpy.place, arr, vals)


@_array_func(numpy.cov)
def cov(m: Quantity, y: Optional[Quantity] = None, rowvar: bool = True, bias: bool = False,
        ddof: Optional[int] = None, fweights: Optional[Quantity] = None,
        aweights: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if y is not None:
        _ = _match_units(numpy.cov, m, y)
    if fweights is not None:
        _ = _scalar_units(numpy.cov, fweights, labels={0: "'fweights'"})
    if aweights is not None:
        _ = _scalar_units(numpy.cov, aweights, labels={0: "'aweights'"})
    return m.dimension**2, m.scale


@_array_func(numpy.corrcoef)
def corrcoeff(x: Quantity, y: Optional[Quantity] = None, rowvar: bool = True) -> Tuple[Dimension, Scale]:
    if y is not None:
        _ = _match_units(numpy.corrcoef, x, y)
    return Scalar, x.scale


@_array_func(numpy.i0)
def i0(x: Quantity) -> Tuple[Dimension, Scale]:
    return _scalar_units(numpy.i0, x)


@_array_func(numpy.sinc)
def sinc(x: Quantity) -> Tuple[Dimension, Scale]:
    return _scalar_units(numpy.sinc, x)


@_array_func(numpy.msort)
def msort(a: Quantity) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.median)
def median(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None, out: Optional[Quantity] = None,
           overwrite_input: bool = False, keepdims: bool = False) -> Tuple[Dimension, Scale]:
    if out is not None:
        return _match_units(numpy.median, a, out, labels={1: "'out'"})
    return a.dimension, a.scale


@_array_func(numpy.percentile)
def percentile(a: Quantity, q: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
               out: Optional[Quantity] = None, overwrite_input: bool = False,
               interpolation: str = 'linear', keepdims: bool = False) -> Tuple[Dimension, Scale]:
    _ = _scalar_units(numpy.percentile, q, offset=1)
    if out is not None:
        return _match_units(numpy.percentile, a, out, labels={1: "'out'"})
    return a.dimension, a.scale


@_array_func(numpy.quantile)
def quantile(a: Quantity, q: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
             out: Optional[Quantity] = None, overwrite_input: bool = False,
             interpolation: str = 'linear', keepdims: bool = False) -> Tuple[Dimension, Scale]:
    _ = _scalar_units(numpy.quantile, q, offset=1)
    if out is not None:
        return _match_units(numpy.quantile, a, out, labels={1: "'out'"})
    return a.dimension, a.scale


@_array_func(numpy.trapz)
def trapz(y: Quantity, x: Optional[Quantity] = None, dx: Optional[Quantity] = None,
          axis: int = -1) -> Tuple[Dimension, Scale]:
    if x is not None:
        return _multiply_units(numpy.trapz, y, x)
    if dx is not None:
        return _multiply_units(numpy.trapz, y, dx)
    return y.dimension, y.scale


@_array_func(numpy.meshgrid)
def meshgrid(*xi: Quantity, copy: bool = True, sparse: bool = False,
             indexing: str = 'xy') -> List[Tuple[Dimension, Scale]]:
    return [(x.quantity, x.scale) for x in xi]


@_array_func(numpy.delete)
def delete(arr: Quantity, obj: Union[slice, int, Sequence[int]],
           axis: Optional[int] = None) -> Tuple[Dimension, Scale]:
    return arr.dimension, arr.scale


@_array_func(numpy.insert)
def insert(arr: Quantity, obj: Union[slice, int, Sequence[int]],
           values: Quantity, axis: Optional[int] = None) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.insert, arr, values, offset=1)


@_array_func(numpy.append)
def append(arr: Quantity, values: Quantity, axis: Optional[int] = None) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.append, arr, values)


@_array_func(numpy.digitize)
def digitize(x: Quantity, bins: Quantity, right: bool = False) -> None:
    _ = _match_units(numpy.digitize, x, bins)
    return None


@_array_func(numpy.broadcast_to)
def broadcast_to(array: Quantity, shape: Sequence[int], subok: bool = False) -> Tuple[Dimension, Scale]:
    return array.dimension, array.scale


@_array_func(numpy.broadcast_arrays)
def broadcast_arrays(*args: Quantity, subok: bool = False) -> List[Tuple[Dimension, Scale]]:
    return [(arr.dimension, arr.scale) for arr in args]


@_array_func(numpy.take_along_axis)
def take_along_axis(arr: Quantity, indices: Array, axis: int) -> Tuple[Dimension, Scale]:
    return arr.dimension, arr.scale


@_array_func(numpy.put_along_axis)
def put_along_axis(arr: Quantity, indices: Array, values: Quantity, axis: int) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.put_along_axis, arr, values, offset=1)


@_array_func(numpy.apply_along_axis)
def apply_along_axis(func1d: Callable, axis: int, arr: Quantity,
                     *args: Any, **kwargs: Any) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.apply_along_axis, func1d(arr[0], *args, **kwargs))


# TODO: think about apply_over_axes


@_array_func(numpy.expand_dims)
def expand_dims(a: Quantity, axis: Union[int, Sequence[int]]) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.column_stack)
def column_stack(tup: Sequence[Quantity]) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.column_stack, *tup)


@_array_func(numpy.dstack)
def dstack(tup: Sequence[Quantity]) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.dstack, *tup)


@_array_func(numpy.array_split)
def array_split(ary: Quantity, indices_or_sections: Union[int, ArrayLike],
                axis: int = 0) -> List[Tuple[Dimension, Scale]]:
    try:
        N = len(cast(Sized, indices_or_sections)) + 1
    except TypeError:
        N = cast(int, indices_or_sections)
    return [(ary.dimension, ary.scale)] * N


@_array_func(numpy.split)
def split(ary: Quantity, indices_or_sections: Union[int, ArrayLike],
          axis: int = 0) -> List[Tuple[Dimension, Scale]]:
    try:
        N = len(cast(Sized, indices_or_sections)) + 1
    except TypeError:
        N = cast(int, indices_or_sections)
    return [(ary.dimension, ary.scale)] * N


@_array_func(numpy.hsplit)
def hsplit(ary: Quantity, indices_or_sections: Union[int, ArrayLike]) -> List[Tuple[Dimension, Scale]]:
    try:
        N = len(cast(Sized, indices_or_sections)) + 1
    except TypeError:
        N = cast(int, indices_or_sections)
    return [(ary.dimension, ary.scale)] * N


@_array_func(numpy.vsplit)
def vsplit(ary: Quantity, indices_or_sections: Union[int, ArrayLike]) -> List[Tuple[Dimension, Scale]]:
    try:
        N = len(cast(Sized, indices_or_sections)) + 1
    except TypeError:
        N = cast(int, indices_or_sections)
    return [(ary.dimension, ary.scale)] * N


@_array_func(numpy.dsplit)
def dsplit(ary: Quantity, indices_or_sections: Union[int, ArrayLike]) -> List[Tuple[Dimension, Scale]]:
    try:
        N = len(cast(Sized, indices_or_sections)) + 1
    except TypeError:
        N = cast(int, indices_or_sections)
    return [(ary.dimension, ary.scale)] * N


@_array_func(numpy.kron)
def kron(a: Quantity, b: Quantity) -> Tuple[Dimension, Scale]:
    return _multiply_units(numpy.kron, a, b)


@_array_func(numpy.tile)
def tile(A: Quantity, reps: ArrayLike) -> Tuple[Dimension, Scale]:
    return A.dimension, A.scale


@_array_func(numpy.ediff1d)
def ediff1d(ary: Quantity, to_end: Optional[Quantity] = None,
            to_begin: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if to_end is None:
        if to_begin is None:
            return ary.dimension, ary.scale
        else:
            return _match_units(numpy.ediff1d, ary, to_begin, labels={1: "'to_begin'"})
    else:
        if to_begin is None:
            return _match_units(numpy.ediff1d, ary, to_end, labels={1: "'to_end'"})
        else:
            return _match_units(numpy.ediff1d, ary, to_end, to_begin, labels={1: "'to_end'", 2: "'to_begin'"})


@_array_func(numpy.unique)
def unique(ar: Quantity, return_index: bool = False, return_inverse: bool = False,
           return_counts: bool = False, axis: Optional[int] = None) -> Tuple[Dimension, Scale]:
    return ar.dimension, ar.scale


@_array_func(numpy.intersect1d)
def intersect1d(ar1: Quantity, ar2: Quantity, assume_unique: bool = False,
                return_indices: bool = False) -> List[Optional[Tuple[Dimension, Scale]]]:
    if return_indices:
        return [_match_units(numpy.intersect1d, ar1, ar2), None, None]
    else:
        return [_match_units(numpy.intersect1d, ar1, ar2)]


@_array_func(numpy.setxor1d)
def setxor1d(ar1: Quantity, ar2: Quantity, assume_unique: bool = False) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.setxor1d, ar1, ar2)


@_array_func(numpy.in1d)
def in1d(ar1: Quantity, ar2: Quantity, assume_unique: bool = False, invert: bool = False) -> None:
    _ = _match_units(numpy.in1d, ar1, ar2)
    return None


@_array_func(numpy.isin)
def isin(element: Quantity, test_elements: Quantity, assume_unique: bool = False, invert: bool = False) -> None:
    _ = _match_units(numpy.isin, element, test_elements)
    return None


@_array_func(numpy.union1d)
def union1d(ar1: Quantity, ar2: Quantity) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.union1d, ar1, ar2)


@_array_func(numpy.setdiff1d)
def setdiff1d(ar1: Quantity, ar2: Quantity, assume_unique: bool = False) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.setdiff1d, ar1, ar2)


@_array_func(numpy.fix)
def fix(x: Quantity, out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return x.dimension, x.scale
    else:
        return _match_units(numpy.fix, x, out, labels={1: "'out'"})


@_array_func(numpy.isposinf)
def isposinf(x: Quantity, out: Optional[Quantity] = None) -> None:
    if out is not None:
        _ = _match_units(numpy.isposinf, x, out, labels={1: "'out'"})
    return None


@_array_func(numpy.isneginf)
def isneginf(x: Quantity, out: Optional[Quantity] = None) -> None:
    if out is not None:
        _ = _match_units(numpy.isneginf, x, out, labels={1: "'out'"})
    return None


@_array_func(numpy.asfarray)
def asfarray(a: Quantity, dtype: Dtype = numpy.float64) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.real)
def real(val: Quantity) -> Tuple[Dimension, Scale]:
    return val.dimension, val.scale


@_array_func(numpy.imag)
def imag(val: Quantity) -> Tuple[Dimension, Scale]:
    return val.dimension, val.scale


@_array_func(numpy.iscomplex)
def iscomplex(x: Quantity) -> None:
    return None


@_array_func(numpy.isreal)
def isreal(x: Quantity) -> None:
    return None


@_array_func(numpy.iscomplexobj)
def iscomplexobj(x: Quantity) -> None:
    return None


@_array_func(numpy.isrealobj)
def isrealobj(x: Quantity) -> None:
    return None


@_array_func(numpy.nan_to_num)
def nan_to_num(val: Quantity, copy: bool = True, nan: Optional[Quantity] = None,
               posinf: Optional[Quantity] = None, neginf: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    args = (nan, posinf, neginf)
    labels = ("'nan'", "'posinf'", "'neginf'")
    return _match_units(
        numpy.nan_to_num, val,
        *(arg for arg in args if arg is not None),
        labels={n + 1: label for n, label in enumerate(label for i, label in enumerate(labels) if args[i] is not None)},
    )


@_array_func(numpy.real_if_close)
def real_if_close(a: Quantity, tol: float = 100.) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.asscalar)
def asscalar(a: Quantity) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.common_type)
def common_type(*arrays: Quantity) -> None:
    return None


@_array_func(numpy.nanmin)
def nanmin(a: Quantity, axis: Optional[int] = None, out: Optional[Quantity] = None,
           keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.nanmin, a, out, labels={1: "'out'"})


@_array_func(numpy.nanmax)
def nanmax(a: Quantity, axis: Optional[int] = None, out: Optional[Quantity] = None,
           keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.nanmax, a, out, labels={1: "'out'"})


@_array_func(numpy.nanargmin)
def nanargmin(a: Quantity, axis: Optional[int] = None) -> None:
    return None


@_array_func(numpy.nanargmax)
def nanargmax(a: Quantity, axis: Optional[int] = None) -> None:
    return None


@_array_func(numpy.nansum)
def nansum(a: Quantity, axis: Optional[int] = None, dtype: Optional[Dtype] = None,
           out: Optional[Quantity] = None, keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.nansum, a, out, labels={1: "'out'"})


@_array_func(numpy.nanprod)
def nanprod(a: Quantity, axis: Optional[int] = None, dtype: Optional[Dtype] = None,
            out: Optional[Quantity] = None, keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.nanprod, a, out, labels={1: "'out'"})


@_array_func(numpy.nancumsum)
def nancumsum(a: Quantity, axis: Optional[int] = None, dtype: Optional[Dtype] = None,
              out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.nancumsum, a, out, labels={1: "'out'"})


@_array_func(numpy.nancumprod)
def nancumprod(a: Quantity, axis: Optional[int] = None, dtype: Optional[Dtype] = None,
               out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.nancumprod, a, out, labels={1: "'out'"})


@_array_func(numpy.nanmean)
def nanmean(a: Quantity, axis: Optional[int] = None, dtype: Optional[Dtype] = None,
            out: Optional[Quantity] = None, keepdims: Any = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.nanmean, a, out, labels={1: "'out'"})


@_array_func(numpy.nanmedian)
def nanmedian(a: Quantity, axis: Optional[int] = None, out: Optional[Quantity] = None,
              overwrie_input: bool = False, keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.nanmedian, a, out, labels={1: "'out'"})


@_array_func(numpy.nanpercentile)
def nanpercentile(a: Quantity, q: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
                  out: Optional[Quantity] = None, overwrite_input: bool = False,
                  interpolation: str = 'linear', keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    _ = _scalar_units(numpy.nanpercentile, q, offset=1)
    if out is not None:
        return _match_units(numpy.nanpercentile, a, out, labels={1: "'out'"})
    return a.dimension, a.scale


@_array_func(numpy.nanquantile)
def nanquantile(a: Quantity, q: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
                out: Optional[Quantity] = None, overwrite_input: bool = False,
                interpolation: str = 'linear', keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    _ = _scalar_units(numpy.nanquantile, q, offset=1)
    if out is not None:
        return _match_units(numpy.nanquantile, a, out, labels={1: "'out'"})
    return a.dimension, a.scale


@_array_func(numpy.nanvar)
def nanvar(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
           dtype: Optional[Dtype] = None, out: Optional[Quantity] = None,
           ddof: int = 0, keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension**2, a.scale
    else:
        dimension, scale = _match_units(numpy.nanvar, a, out, labels={1: "'out'"})
        return dimension**2, scale


@_array_func(numpy.nanstd)
def nanstd(a: Quantity, axis: Optional[Union[int, Sequence[int]]] = None,
           dtype: Optional[Dtype] = None, out: Optional[Quantity] = None,
           ddof: int = 0, keepdims: Optional[bool] = None) -> Tuple[Dimension, Scale]:
    if out is None:
        return a.dimension, a.scale
    else:
        return _match_units(numpy.nanstd, a, out, labels={1: "'out'"})


@_array_func(numpy.pad)
def pad(array: Quantity, pad_width: Union[int, Sequence[int], ArrayLike],
        mode: Union[str, Callable] = 'constant', stat_length: Optional[Union[int, Sequence[int]]] = None,
        constant_values: Optional[Quantity] = None, end_values: Optional[Quantity] = None,
        reflect_type: Optional[str] = None) -> Tuple[Dimension, Scale]:
    args: Tuple[Quantity, ...] = ()
    labels: Tuple[str, ...] = ()
    if constant_values is not None:
        args += (constant_values,)
        labels += ("'constant_values'",)
    if end_values is not None:
        args += (end_values,)
        labels += ("'end_values'",)
    if len(args) == 0:
        return array.dimension, array.scale
    else:
        return _match_units(numpy.pad, array, *args, labels={i + 1: name for i, name in enumerate(labels)})


@_array_func(numpy.linalg.tensorsolve)
def tensorsolve(a: Quantity, b: Quantity, axes: Optional[Sequence[int]] = None) -> Tuple[Dimension, Scale]:
    return _divide_units(numpy.linalg.tensorsolve, b, a, labels={0: "second", 1: "first"})


@_array_func(numpy.linalg.solve)
def solve(a: Quantity, b: Quantity) -> Tuple[Dimension, Scale]:
    return _divide_units(numpy.linalg.solve, b, a, labels={0: "second", 1: "first"})


@_array_func(numpy.linalg.tensorinv)
def tensorinv(a: Quantity, ind: int = 2) -> Tuple[Dimension, Scale]:
    return Scalar / a.dimension, a.scale


@_array_func(numpy.linalg.inv)
def inv(a: Quantity) -> Tuple[Dimension, Scale]:
    return Scalar / a.dimension, a.scale


@_array_func(numpy.linalg.matrix_power)
def matrix_power(a: Quantity, n: int) -> Tuple[Dimension, Scale]:
    return a.dimension ** n, a.scale


@_array_func(numpy.linalg.cholesky)
def cholesky(a: Quantity) -> Tuple[Dimension, Scale]:
    return a.dimension ** 0.5, a.scale


@_array_func(numpy.linalg.qr)
def qr(a: Quantity, mode: str = 'reduced') -> List[Optional[Tuple[Dimension, Scale]]]:
    if mode == 'reduced' or mode == 'complete':
        return [(Scalar, a.scale), (a.dimension, a.scale)]
    if mode == 'r':
        return [(a.dimension, a.scale)]
    if mode == 'raw' and a.dimension == Scalar:
        return [None]
    raise ValueError(f"Mode '{mode}' not supported for QR decompositions of Quantities")


@_array_func(numpy.linalg.eigvals)
def eigvals(a: Quantity) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.linalg.eigvalsh)
def eigvalsh(a: Quantity, UPLO: str = 'L') -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.linalg.eig)
def eig(a: Quantity) -> List[Optional[Tuple[Dimension, Scale]]]:
    return [(a.dimension, a.scale), None]


@_array_func(numpy.linalg.eigh)
def eigh(a: Quantity, UPLO: str = 'L') -> List[Optional[Tuple[Dimension, Scale]]]:
    return [(a.dimension, a.scale), None]


@_array_func(numpy.linalg.svd)
def svd(a: Quantity, full_matrices: bool = True, compute_uv: bool = True,
        hermitian: bool = False) -> List[Tuple[Dimension, Scale]]:
    return [(Scalar, a.scale), (a.dimension, a.scale), (Scalar, a.scale)]


@_array_func(numpy.linalg.cond)
def cond(x: Quantity, p: Optional[Union[int, str]] = None) -> None:
    return None


@_array_func(numpy.linalg.matrix_rank)
def matrix_rank(M: Quantity, tol: Optional[Quantity] = None, hermitian: bool = False) -> None:
    if tol is not None:
        _ = _match_units(numpy.linalg.matrix_rank, M, tol)
    return None


@_array_func(numpy.linalg.pinv)
def pinv(a: Quantity, rcond: Optional[Quantity] = None, hermitian: bool = False) -> Tuple[Dimension, Scale]:
    if rcond is not None:
        _ = _scalar_units(numpy.linalg.pinv, rcond, labels={0: "'rcond'"})
    return Scalar / a.dimension, a.scale


@_array_func(numpy.linalg.slogdet)
def slogdet(a: Quantity) -> List[Optional[Tuple[Dimension, Scale]]]:
    return [None, _scalar_units(numpy.linalg.slogdet, a)]


@_array_func(numpy.linalg.det)
def det(a: Quantity) -> Tuple[Dimension, Scale]:
    M = numpy.shape(a)[-1]
    return a.dimension**M, a.scale


@_array_func(numpy.linalg.lstsq)
def lstsq(a: Quantity, b: Quantity, rcond: Optional[Quantity] = None) -> List[Optional[Tuple[Dimension, Scale]]]:
    if rcond is not None:
        _ = _scalar_units(numpy.linalg.lstsq, rcond, labels={0: "'rcond'"})
    return [
        _divide_units(numpy.linalg.lstsq, b, a, labels={0: "second", 1: "first"}),
        (b.dimension, b.scale),
        None,
        (a.dimension, a.scale),
    ]


@_array_func(numpy.linalg.norm)
def norm(x: Quantity, ord: Optional[Union[int, str]] = None,
         axis: Optional[Union[int, Tuple[int, int]]] = None,
         keepdims: bool = False) -> Tuple[Dimension, Scale]:
    return x.dimension, x.scale


@_array_func(numpy.linalg.multi_dot)
def multi_dot(arrays: Sequence[Quantity], *, out: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    return _multiply_units(numpy.linalg.multi_dot, *arrays, argument="array")


@_array_func(numpy.fft.fftshift)
def fftshift(x: Quantity, axes: Optional[Union[int, Sequence[int]]] = None) -> Tuple[Dimension, Scale]:
    return x.dimension, x.scale


@_array_func(numpy.fft.ifftshift)
def ifftshift(x: Quantity, axes: Optional[Union[int, Sequence[int]]] = None) -> Tuple[Dimension, Scale]:
    return x.dimension, x.scale


@_array_func(numpy.fft.fft)
def fft(a: Quantity, n: Optional[int] = None, axis: int = -1,
        norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.ifft)
def ifft(a: Quantity, n: Optional[int] = None, axis: int = -1,
         norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.rfft)
def rfft(a: Quantity, n: Optional[int] = None, axis: int = -1,
         norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.irfft)
def irfft(a: Quantity, n: Optional[int] = None, axis: int = -1,
          norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.hfft)
def hfft(a: Quantity, n: Optional[int] = None, axis: int = -1,
         norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.ihfft)
def ihfft(a: Quantity, n: Optional[int] = None, axis: int = -1,
          norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.fftn)
def fftn(a: Quantity, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None,
         norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.ifftn)
def ifftn(a: Quantity, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None,
          norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.fft2)
def fft2(a: Quantity, s: Optional[Sequence[int]] = None, axes: Sequence[int] = (-2, -1),
         norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.ifft2)
def ifft2(a: Quantity, s: Optional[Sequence[int]] = None, axes: Sequence[int] = (-2, -1),
          norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.rfftn)
def rfftn(a: Quantity, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None,
          norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.irfftn)
def irfftn(a: Quantity, s: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None,
           norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.rfft2)
def rfft2(a: Quantity, s: Optional[Sequence[int]] = None, axes: Sequence[int] = (-2, -1),
          norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.fft.irfft2)
def irfft2(a: Quantity, s: Optional[Sequence[int]] = None, axes: Sequence[int] = (-2, -1),
           norm: Optional[str] = None) -> Tuple[Dimension, Scale]:
    return a.dimension, a.scale


@_array_func(numpy.ix_)
def ix_(*args: Quantity) -> List[Tuple[Dimension, Scale]]:
    return [(a.dimension, a.scale) for a in args]


@_array_func(numpy.fill_diagonal)
def fill_diagonal(a: Quantity, val: Quantity, wrap: bool = False) -> None:
    _ = _match_units(numpy.fill_diagonal, a, val)
    return None


@_array_func(numpy.diag_indices_from)
def diag_indices_from(arr: Quantity) -> None:
    return None


@_array_func(numpy.save)
def save(file: Union[BinaryIO, str, Path], arr: Quantity,
         allow_pickle: bool = True, fix_imports: bool = True) -> None:
    _ = _scalar_units(numpy.save, arr)
    return None


@_array_func(numpy.savez)
def savez(file: Union[BinaryIO, str, Path], *args: Quantity, **kwds: Quantity) -> None:
    _ = _scalar_units(numpy.savez, *args, *kwds.values(),
                      labels={len(args) + i: f"'{key}'" for i, key in enumerate(kwds.keys())})
    return None


@_array_func(numpy.savez_compressed)
def savez_compressed(file: Union[BinaryIO, str, Path], *args: Quantity, **kwds: Quantity) -> None:
    _ = _scalar_units(numpy.savez_compressed, *args, *kwds.values(),
                      labels={len(args) + i: f"'{key}'" for i, key in enumerate(kwds.keys())})
    return None


@_array_func(numpy.savetxt)
def savetxt(fname: Union[TextIO, str, Path], X: Quantity, fmt: Union[str, Sequence[str]] = '%.18e',
            delimiter: str = ' ', newline: str = '\n', header: str = '',
            footer: str = '', comments: str = '# ', encoding: Optional[str] = None) -> None:
    _ = _scalar_units(numpy.savetxt, X)
    return None


@_array_func(numpy.poly)
def poly(seq_of_zeros: Quantity) -> Tuple[Dimension, Scale]:
    return _scalar_units(numpy.poly, seq_of_zeros)


@_array_func(numpy.roots)
def roots(p: Quantity) -> Tuple[Dimension, Scale]:
    return Scalar, p.scale


@_array_func(numpy.polyint)
def polyint(p: Quantity, m: int = 1, k: Optional[Quantity] = None) -> Tuple[Dimension, Scale]:
    if k is not None:
        return _match_units(numpy.polyint, p, k, offset=1)
    return p.dimension, p.scale


@_array_func(numpy.polyder)
def polyder(p: Quantity, m: int = 1) -> Tuple[Dimension, Scale]:
    return p.dimension, p.scale


@_array_func(numpy.polyfit)
def polyfit(x: Quantity, y: Quantity, deg: int, rcond: Optional[float] = None,
            full: bool = False, w: Optional[Quantity] = None,
            cov: Union[bool, str] = False) -> Union[Tuple[Dimension, Scale], List[Optional[Tuple[Dimension, Scale]]]]:
    _ = _scalar_units(numpy.polyfit, x)
    if w is not None:
        prod = _multiply_units(numpy.polyfit, y, w, labels={1: "w"})
    else:
        prod = (y.dimension, y.scale)
    if full:
        # TODO: check that singular values should be scalar
        return [(y.dimension, y.scale), (prod[0]**2, prod[1]), None, (Scalar, y.scale), None]
    if cov:
        return [(y.dimension, y.scale), (y.dimension**2, y.scale)]
    return y.dimension, y.scale


@_array_func(numpy.polyval)
def polyval(p: Quantity, x: Quantity) -> Tuple[Dimension, Scale]:
    _ = _scalar_units(numpy.polyval, x)
    return p.dimension, p.scale


@_array_func(numpy.polyadd)
def polyadd(a1: Quantity, a2: Quantity) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.polyadd, a1, a2)


@_array_func(numpy.polysub)
def polysub(a1: Quantity, a2: Quantity) -> Tuple[Dimension, Scale]:
    return _match_units(numpy.polysub, a1, a2)


@_array_func(numpy.polymul)
def polymul(a1: Quantity, a2: Quantity) -> Tuple[Dimension, Scale]:
    return _multiply_units(numpy.polymul, a1, a2)


@_array_func(numpy.polydiv)
def polydiv(a1: Quantity, a2: Quantity) -> Tuple[Dimension, Scale]:
    return _divide_units(numpy.polydiv, a1, a2)


# Special handling for block method due to nested sequences
# TODO: implement this

# def _flatten_block(arrays: Any) -> Iterable[Any]:
#     if type(arrays) is list:
#         for el in arrays:
#             yield from _flatten_block(el)


# @_block_func(numpy.block)
# def block(arrays: Sequence[Any]) -> Tuple[Dimension, Scale]:
#     flattened = tuple(_flatten_block(arrays))
#
#     try:
#         scale = next(iter(value.scale for value in flattened if isinstance(value, Quantity)))
#     except StopIteration:
#         raise TypeError(
#             f"Invalid argument for '{numpy.block.__name__}': unexpected Quantity"
#         )
#
#     wrapped = (value if isinstance(value, Quantity) else Quantity(value, Scalar, scale) for value in flattened)
#
#     return _match_units(numpy.block, *wrapped, argument="block")


# Special handling for einsum methods due to alternate calling convention

@_einsum_func(numpy.einsum_path)
def einsum_path(args: Sequence[Quantity], labels: Mapping[int, str]) -> Tuple[Dimension, Scale]:
    return _multiply_units(numpy.einsum_path, *args, labels=labels, offset=1)


@_einsum_func(numpy.einsum)
def einsum(args: Sequence[Quantity], labels: Mapping[int, str]) -> Tuple[Dimension, Scale]:
    return _multiply_units(numpy.einsum, *args, labels=labels, offset=1)

# TODO: add any relevant functions from
# numpy/lib/histograms.py
# numpy/lib/recfunctions.py
# numpy/lib/scimath.py
# numpy/lib/twodim_base.py
