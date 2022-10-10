from abc import ABC, abstractmethod
from functools import reduce, wraps
import functools
from iminuit import Minuit, describe  # type: ignore
import iminuit.cost  # type: ignore
import numpy
from xarray import Dataset, DataArray
from typing import Callable, Hashable, Union, Mapping, Sequence, Any, Tuple, Type, Optional, Iterator, List, Iterable, Literal, Dict
import itertools
from dataclasses import dataclass, field
from operator import mul
import pandas  # type: ignore
from numba import jit  # type: ignore

from ilaff.units import Quantity, QuantityIndex, Scalar, one, in_unit, Physical
from ilaff.units.quantity import _upcast_types

from ilaff.fit import IntoModel, Model, ModelFn, PartialModel, partial, jackknife, value_jack, error_jack, covariance_jack, bootstrap, value_boot, error_boot, covariance_boot


class Cost(iminuit.cost.Cost):
    def __init__(self, var: DataArray, model: DataArray, value: Callable[[DataArray], DataArray], covariance: Callable[[DataArray, DataArray], DataArray], covariant_dims: Sequence[str] = (), verbose: int = 0):
        pass

    def set_function(self, select: Callable[[DataArray], DataArray], value: Callable[[DataArray], DataArray], covariance: Callable[[DataArray, DataArray], DataArray]) -> None:
        pass


# class NDCorrelatedChiSquared(Cost):
#     def __init__(self, var: DataArray, model: DataArray, value: Callable[[DataArray], DataArray], covariance: Callable[[DataArray, DataArray], DataArray], covariant_dims: Sequence[str] = (), verbose: int = 0):
#         self.residual = var - model
#         self.covariant_dims = covariant_dims
# 
#         self.set_function(value, covariance)
# 
#         args = describe(self.fn)
# 
#         iminuit.cost.Cost.__init__(self, args, len(self.residual), verbose)
# 
#     def set_function(self, value: Callable[[DataArray], DataArray], covariance: Callable[[DataArray, DataArray], DataArray]) -> None:
#         if len(self.covariant_dims) == 0:
#             r = value(self.residual)
#             v = covariance(self.residual, self.residual)
# 
#             print(r.dims, v.dims)
# 
#             chi2 = (r**2 / v).sum()
#             self.chi2 = in_unit(chi2, one).data
#             self.fn = in_unit(chi2, one).data.simplify().compile()
#         else:
#             r = self.residual.stack(covariant_dims=covariant_dims)
#             rbar = value(r.expand_dims('covariant_dims_2', -2))
#             r = value(r.expand_dims('covariant_dims_2', -1))
#             cov = covariance(r, rbar)
#             inv_cov = numpy.linalg.inv(cov.data)
#             chi2 = numpy.sum(rbar.data @ inv_cov @ r.data)
#             self.chi2 = in_unit(chi2, one).data
#             self.fn = in_unit(chi2, one).data.simplify().compile()
# 
#     def _call(self, args) -> float:
#         return self.fn(*args)


class CorrelatedChiSquared(Cost):
    def __init__(self, var: DataArray, model: DataArray, value: Callable[[DataArray], DataArray], covariance: Callable[[DataArray, DataArray], DataArray], covariant_dims: Sequence[str] = (), verbose: int = 0):
        self.var = var
        self.model = model
        self.covariant_dims = covariant_dims

        self.set_function(lambda v: v, value, covariance)

        args = describe(self.fn)

        iminuit.cost.Cost.__init__(self, args, verbose)

    def _call(self, args) -> float:
        return self.fn(*args)

    @property
    def ndata(self) -> int:
        return len(self.var - self.model)


class NDCorrelatedChiSquared(CorrelatedChiSquared):
    def set_function(self, select: Callable[[DataArray], DataArray], value: Callable[[DataArray], DataArray], covariance: Callable[[DataArray, DataArray], DataArray]) -> None:
        residual_pfunc = select(self.var - self.model)
        residual_compiled = residual_pfunc.data.value.simplify().compile()
        if len(self.covariant_dims) == 0:
            @wraps(residual_compiled)
            def chi2(*args):
                res = residual_pfunc.copy(data=Quantity(
                    residual_compiled(*args),
                    residual_pfunc.data.dimension,
                    residual_pfunc.data.scale,
                ))

                r = value(res)
                v = covariance(res, res)

                chi2 = (r**2 / v).sum()
                return in_unit(chi2, one).data
            self.fn = chi2
        else:
            @wraps(residual_compiled)
            def chi2(*args):
                res = residual_pfunc.copy(data=Quantity(
                    residual_compiled(*args),
                    residual_pfunc.data.dimension,
                    residual_pfunc.data.scale,
                ))

                r = res.stack(covariant_dims=self.covariant_dims)
                rbar = r.rename(covariant_dims='covariant_dims_2')
                cov = covariance(r, rbar)
                rbar = rbar.expand_dims('covariant_dims', -2)
                r = r.expand_dims('covariant_dims_2', -1)
                try:
                    inv_cov = numpy.linalg.inv(cov.data.value)
                except:
                    print(args)
                    print(r.data.value)
                    print(cov.data.value)
                    raise
                r, rbar = (value(r).data.value, value(rbar).data.value)
                chi2 = numpy.sum(rbar @ inv_cov @ r)
                return chi2
            self.fn = chi2


class SimpleCorrelatedChiSquared(CorrelatedChiSquared):
    def set_function(self, select: Callable[[DataArray], DataArray], value: Callable[[DataArray], DataArray], covariance: Callable[[DataArray, DataArray], DataArray]) -> None:
        residual_pfunc = value(select(self.var - self.model))
        residual_compiled = residual_pfunc.data.value.simplify().compile()
        if len(self.covariant_dims) == 0:
            var = select(self.var)
            v = covariance(var, var)

            @wraps(residual_compiled)
            def chi2(*args):
                r = residual_pfunc.copy(data=Quantity(
                    residual_compiled(*args),
                    residual_pfunc.data.dimension,
                    residual_pfunc.data.scale,
                ))

                chi2 = (r**2 / v).sum()
                return in_unit(chi2, one).data
            self.fn = chi2
        else:
            v = select(self.var).stack(covariant_dims=self.covariant_dims)
            vbar = v.rename(covariant_dims='covariant_dims_2')
            cov = covariance(v, vbar)
            try:
                inv_cov = numpy.linalg.inv(cov.data.value)
            except:
                print(cov.data.value)
                raise

            @wraps(residual_compiled)
            def chi2(*args):
                res = residual_pfunc.copy(data=Quantity(
                    residual_compiled(*args),
                    residual_pfunc.data.dimension,
                    residual_pfunc.data.scale,
                ))

                r = res.stack(covariant_dims=self.covariant_dims)
                rbar = r.rename(covariant_dims='covariant_dims_2')
                rbar = rbar.expand_dims('covariant_dims', -2).data.value
                r = r.expand_dims('covariant_dims_2', -1).data.value
                chi2 = numpy.sum(rbar @ inv_cov @ r)
                return chi2
            self.fn = chi2


def unwrap_xarray(a: Any) -> Any:
    variable = getattr(a, 'variable', None)
    if variable is None:
        return a
    data = variable._data
    if getattr(data, "shape", None) == ():
        return data[()]
    return getattr(data, "array", data)


def check_model_units(data: Dataset, var: DataArray, model: IntoModel, kwargs: Mapping[str, Any]) -> Tuple[Mapping[str, Quantity], Mapping[str, Any], Mapping[str, Any]]:
    data = data.isel({dim: 0 for dim in data.dims})
    y = unwrap_xarray(var)

    scale = next(
        (
            q.scale for q in itertools.chain(
                (y,),
                (unwrap_xarray(q) for q in itertools.chain(
                    data.values(),
                    data.coords.values(),
                )),
                (unwrap_xarray(q[0] if isinstance(q, tuple) else q) for q in kwargs.values())
            ) if isinstance(q, Quantity)
        ),
        Physical(),
    )

    model = Model.new(model)
    sig = describe(model)

    params = {
        v: q if isinstance(q, Quantity) else Quantity(q, Scalar, scale)
        for v, q in (
            (v, next((
                unwrap_xarray(q[0] if isinstance(q, tuple) else q)
                for q in (kwargs.get(f"{prefix}{v}")
                          for prefix in ("", "error_", "limit_", "fix_"))
                if q is not None
            ), None))
            for v in sig
        )
        if q is not None
    }

    if scale is None:
        return (
            {k: 1 for k in sig},
            {k: v for k, v in kwargs.items() if k in params},
            {k[6:]: v for k, v in kwargs.items() if k[:6] == "limit_" and k[6:] in params},
        )

    def check_unwrap(k: str, v: Quantity) -> Any:
        q: Optional[Quantity] = None
        for prefix in ("", "error_", "limit_", "fix_"):
            if k.startswith(prefix) and k[len(prefix):] in params:
                q = params[k[len(prefix):]]
                break
        if q is None:
            raise ValueError(f"Quantity passed to unexpected argument '{k}'")
        if v.dimension != q.dimension:
            raise ValueError(
                f"Invalid mass dimension for argument '{k}': got {v.dimension}, expected {q.dimension}"
            )
        if v.dimension != Scalar and v.scale != q.scale:
            raise ValueError(
                f"Mismatched scale for argument '{k}'"
            )
        return v.value

    def check_scalar(k: str, v: Any) -> Any:
        q: Optional[Quantity] = None
        for prefix in ("", "error_", "limit_", "fix_"):
            if k.startswith(prefix) and k[len(prefix):] in params:
                q = params[k[len(prefix):]]
                break
        if q is not None:
            if Scalar != q.dimension:
                raise ValueError(
                    f"Invalid mass dimension for argument '{k}': got {Scalar}, expected {q.dimension}"
                )
        return v

    unwrapped = {
        k: check_unwrap(k, v) if isinstance(v, Quantity)
           else tuple(check_unwrap(k, e) if isinstance(e, Quantity) else check_scalar(k, e) for e in v) if isinstance(v, tuple)
           else check_scalar(k, v)
        for k, v in ((k, tuple(unwrap_xarray(e) for e in v) if isinstance(v, tuple) else unwrap_xarray(v)) for k, v in kwargs.items())
    }

    args = [unwrap_xarray(data.get(v, params.get(v))) for v in sig]
    for v, p in zip(sig, args):
        if p is None:
            raise ValueError(f"Model parameter '{v}' not specified")
    return (
        {k: Quantity(1, v.dimension, v.scale) for k, v in params.items()},
        {k: v for k, v in unwrapped.items() if k in params},
        {k[6:]: v for k, v in unwrapped.items() if k[:6] == "limit_" and k[6:] in params},
    )


ArrayLike = Any
DType = Any


# TODO: fix Placeholder * DataArray does not work when not using Quantities
@dataclass(frozen=True)  # type: ignore
class PartialExpr(numpy.lib.mixins.NDArrayOperatorsMixin, pandas.api.extensions.ExtensionArray, ABC):
    shape: Tuple[int, ...] = ()

    def locals(self) -> Dict[str, Any]:
        return {}

    def params(self) -> List[str]:
        return []

    @abstractmethod
    def expr(self) -> str:
        pass

    @abstractmethod
    def __getitem__(self, key: Any) -> "PartialExpr":
        pass

    @abstractmethod
    def _distribute_func(self, func: Callable, args: Sequence[Any], kwargs: Mapping[str, Any]) -> "PartialExpr":
        pass

    @abstractmethod
    def _arrays(self) -> Iterable[ArrayLike]:
        pass

    @abstractmethod
    def astype(self, dtype: DType, order: str = 'K', casting: str = 'unsafe',
               subok: bool = True, copy: bool = True) -> "Quantity":
        pass

    @abstractmethod
    def _placeholder(self) -> Any:
        pass

    @abstractmethod
    def simplify(self) -> "PartialExpr":
        pass

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return numpy.broadcast(*self._arrays()).size

    def transpose(self, axes: Sequence[int]) -> "PartialExpr":
        return numpy.transpose(self, axes)  # type: ignore

    def take(self, indices: ArrayLike, axis: Optional[int] = None,
             out: Optional["PartialExpr"] = None, mode: Union[Literal['raise'], Literal['wrap'], Literal['clip']] = 'raise') -> "PartialExpr":
        if out is not None:
            raise ValueError("Cannot modify PartialExpr in-place")
        return numpy.take(self, indices, axis, out, mode)

    def copy(self, order: str = 'K') -> "PartialExpr":
        return numpy.copy(self, order)

    @property
    def dtype(self) -> DType:
        return numpy.dtype(numpy.result_type(self))

    @property
    def nbytes(self) -> int:
        return len(self) * 4

    @staticmethod
    def _from_sequence(scalars: Sequence[Any], dtype: Optional[DType] = None, copy: bool = False) -> "PartialExpr":
        if copy:
            value = numpy.fromiter((s.copy() for s in scalars), dtype=dtype, count=len(scalars))
        else:
            value = numpy.fromiter((s for s in scalars), dtype=dtype, count=len(scalars))
        return ValueExpr(value)

    @classmethod
    def _from_factorized(cls, values: numpy.ndarray, original: "PartialExpr") -> "PartialExpr":
        try:
            arr = original.__array__()
        except ValueError:
            raise NotImplementedError("Can't factorize a PartialExpr with free parameters")
        return ValueExpr(arr[values])

    def _values_for_factorize(self) -> Tuple[numpy.ndarray, float]:
        return numpy.array(self), numpy.nan

    def isna(self):
        return numpy.isnan(self)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence["PartialExpr"]) -> "PartialExpr":
        return numpy.concatenate(to_concat)

    def __bool__(self) -> bool:
        try:
            arr = self.__array__()
        except ValueError:
            raise ValueError("The truth value of a PartialExpr with free parameters is ambiguous")
        return bool(arr)

    @staticmethod
    def _wrap(value: Any) -> "PartialExpr":
        if isinstance(value, PartialExpr):
            return value
        return ValueExpr(value)

    def __array__(self, t=None) -> numpy.ndarray:
        if len(self.params()) > 0:
            raise ValueError("Cannot convert PartialExpr with free parameters into concrete array")
        return eval(self.expr(), self.locals())

    # TODO: think about non-commuting operations
    _upcast_types = _upcast_types + [Quantity, QuantityIndex]
    _distribute_array_funcs = frozenset((
        numpy.take, numpy.reshape, numpy.repeat,
        numpy.swapaxes, numpy.transpose, numpy.resize, numpy.squeeze,
        numpy.diagonal, numpy.ravel, numpy.compress, numpy.concatenate,
        numpy.roll, numpy.rollaxis, numpy.moveaxis, numpy.vstack,
        numpy.hstack, numpy.stack, numpy.rot90, numpy.flip, numpy.copy,
        numpy.broadcast_to, numpy.broadcast_arrays, numpy.take_along_axis,
        numpy.expand_dims, numpy.column_stack, numpy.dstack,
        numpy.array_split, numpy.split, numpy.hsplit, numpy.vsplit,
        numpy.dsplit, numpy.tile, numpy.asscalar,
    ))
    _forbidden_array_funcs = frozenset((
        numpy.min_scalar_type, numpy.copyto, numpy.put,
        numpy.putmask, numpy.shares_memory, numpy.may_share_memory,
        numpy.place, numpy.delete, numpy.insert, numpy.append,
        numpy.put_along_axis, numpy.fill_diagonal, numpy.save, numpy.savez,
        numpy.savez_compressed, numpy.savetxt,
    ))
    _representative_array_funcs = frozenset((
        numpy.ones_like, numpy.zeros_like, numpy.full_like,
        numpy.diag_indices_from,
    ))
    _reduce_array_funcs = frozenset((
        numpy.common_type, numpy.result_type,
    ))
    # _other_array_funcs = frozenset((
    #     numpy.partition, numpy.argpartition, numpy.sort,
    #     numpy.argsort, numpy.argmax, numpy.argmin, numpy.searchsorted,
    #     numpy.trace, numpy.nonzero, numpy.clip, numpy.sum, numpy.any,
    #     numpy.all, numpy.cumsum, numpy.ptp, numpy.amax, numpy.amin,
    #     numpy.prod, numpy.cumprod, numpy.around, numpy.mean, numpy.std,
    #     numpy.var, numpy.round, numpy.product, numpy.cumproduct,
    #     numpy.sometrue, numpy.alltrue, numpy.linspace, numpy.logspace,
    #     numpy.geomspace, numpy.empty_like, numpy.inner, numpy.lexsort,
    #     numpy.dot, numpy.vdot, numpy.choose, numpy.where, numpy.count_nonzero,
    #     numpy.argwhere, numpy.flatnonzero, numpy.correlate, numpy.convolve,
    #     numpy.outer, numpy.tensordot, numpy.cross, numpy.allclose,
    #     numpy.isclose, numpy.array_equal, numpy.array_equiv,
    #     numpy.atleast_1d, numpy.atleast_2d, numpy.atleast_3d,
    #     numpy.average, numpy.piecewise, numpy.select, numpy.gradient,
    #     numpy.diff, numpy.interp, numpy.angle, numpy.unwrap,
    #     numpy.sort_complex, numpy.trim_zeros, numpy.extract, numpy.cov,
    #     numpy.corrcoeff, numpy.i0, numpy.sinc, numpy.msort, numpy.median,
    #     numpy.percentile, numpy.quantile, numpy.trapz, numpy.meshgrid,
    #     numpy.digitize, numpy.apply_along_axis, numpy.kron, numpy.ediff1d,
    #     numpy.unique, numpy.intersect1d, numpy.setxor1d, numpy.in1d,
    #     numpy.isin, numpy.union1d, numpy.fix, numpy.asfarray,
    #     numpy.isposinf, numpy.isneginf, numpy.real, numpy.imag,
    #     numpy.iscomplex, numpy.isreal, numpy.iscomplexobj, numpy.isrealobj,
    #     numpy.nan_to_num, numpy.real_if_close, numpy.nanmin, numpy.nanmax,
    #     numpy.nanargmin, numpy.nanargmax, numpy.nansum, numpy.nanprod,
    #     numpy.nancumsum, numpy.nancumprod, numpy.nanmean, numpy.nanpercentile,
    #     numpy.nanquantile, numpy.nanvar, numpy.nanstd, numpy.pad,
    #     numpy.tensorsolve, numpy.solve, numpy.tensorinv, numpy.inv,
    #     numpy.matrix_power, numpy.cholesky, numpy.qr, numpy.eigvals,
    #     numpy.eigvalsh, numpy.eig, numpy.svd, numpy.cond, numpy.matrix_rank,
    #     numpy.pinv, numpy.slogdet, numpy.det, numpy.lstsq, numpy.norm,
    #     numpy.multi_dot, numpy._ix
    # ))

    def __array_function__(self, func: Callable, types: Iterable[Type],
                           args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
        if any((t in self._upcast_types) for t in types):
            return NotImplemented
        if func in self._forbidden_array_funcs:
            return NotImplemented
        if func is numpy.shape:
            return self.shape
        if func is numpy.ndim:
            return self.ndim
        if func is numpy.size:
            return self.size
        if func in self._reduce_array_funcs:
            assert len(kwargs) == 0
            arrs = [
                a
                for arg in args
                for a in (
                    arg._arrays() if isinstance(arg, PartialExpr) else (arg,)
                )
            ]
            if len(arrs) == 0:
                # TODO: think about if this is necessary
                return func(0)
            return func(*arrs)
        if func in self._representative_array_funcs:
            def rep(arg: Any) -> Any:
                return (next(iter(arg._arrays()), numpy.array(0.))
                        if isinstance(arg, PartialExpr) else (arg,))
            return func(
                *(rep(arg) for arg in args),
                **{k: rep(v) for k, v in kwargs.items()},
            )
        if "out" in kwargs:
            raise ValueError("Cannot modify PartialExpr in-place")
        if func in self._distribute_array_funcs:
            return self._distribute_func(func, args, kwargs)
        return FuncExpr(
            ValueExpr(func),
            tuple(self._wrap(arg) for arg in args),
            {k: self._wrap(arg) for k, arg in kwargs.items()},
        )

    def __array_ufunc__(self, ufunc: numpy.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if any((type(arg) in self._upcast_types) for arg in itertools.chain(inputs, kwargs.values())):
            return NotImplemented
        return UFuncExpr(
            ValueExpr(ufunc),
            tuple(self._wrap(arg) for arg in inputs),
            {k: self._wrap(arg) for k, arg in kwargs.items()},
            method,
        )

    # TODO: delegate relevant methods to numpy

    def compile(self) -> Callable:
        local = self.locals()
        exec(
            (
                f"def fn({', '.join(dict.fromkeys(self.params()))}):\n"
                f"    return {self.expr()}\n"
            ),
            local,
        )
        return local['fn']

    def compile_numba(self) -> Callable:
        local = {
            'jit': jit,
            **self.locals()
        }
        exec(
            (
                "@jit\n"
                f"def fn({', '.join(dict.fromkeys(self.params()))}):\n"
                f"    return {self.expr()}\n"
            ),
            local,
        )
        return local['fn']


class ZeroExpr(PartialExpr):
    def expr(self) -> str:
        return '0.0'

    def __getitem__(self, key: Any) -> "PartialExpr":
        return self

    def _distribute_func(self, func: Callable, args: Sequence[Any], kwargs: Mapping[str, Any]) -> "PartialExpr":
        return ZeroExpr()

    def _arrays(self) -> Iterable[ArrayLike]:
        return ()

    def astype(self, dtype: DType, order: str = 'K', casting: str = 'unsafe',
               subok: bool = True, copy: bool = True) -> "Quantity":
        return ZeroExpr()

    def _placeholder(self) -> Any:
        return 0.0

    def simplify(self) -> "PartialExpr":
        return self


@dataclass(frozen=True, init=False)
class Placeholder(PartialExpr):
    label: str = ""

    def __init__(self, label: str):
        object.__setattr__(self, 'label', label)
        super().__init__()

    def _placeholder(self) -> Any:
        return 0.0

    def params(self) -> List[str]:
        return [self.label]

    def expr(self) -> str:
        return self.label

    def __getitem__(self, key: Any) -> "PartialExpr":
        return self

    def _distribute_func(self, func: Callable, args: Sequence[Any], kwargs: Mapping[str, Any]) -> "PartialExpr":
        return Placeholder(self.label)

    def _arrays(self) -> Iterable[ArrayLike]:
        return ()

    def astype(self, dtype: DType, order: str = 'K', casting: str = 'unsafe',
               subok: bool = True, copy: bool = True) -> "Quantity":
        return Placeholder(self.label)

    def simplify(self) -> "PartialExpr":
        return self


@dataclass(frozen=True, init=False)
class ValueExpr(PartialExpr):
    value: Any = 0

    def __init__(self, value: Any):
        object.__setattr__(self, 'value', value)
        super().__init__(numpy.shape(value))

    def _placeholder(self) -> Any:
        return self.value

    def locals(self) -> Dict[str, Any]:
        return {f"val_{id(self.value)}": self.value}

    def expr(self) -> str:
        return f"val_{id(self.value)}"

    def __getitem__(self, key: Any) -> "PartialExpr":
        if self.ndim == 0:
            return self
        return ValueExpr(self.value[key])

    def _distribute_func(self, func: Callable, args: Sequence[Any], kwargs: Mapping[str, Any]) -> "PartialExpr":
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, PartialExpr):
                if (not arg.__class__ is self.__class__) or numpy.ndim(arg.value) != numpy.ndim(self.value):
                    raise TypeError(f"Inconsistent PartialExprs passed to {func.__name__}")
        # TODO: check this always gets the right ValueExpr
        if self.ndim == 0:
            return self
        return ValueExpr(func(
            *((arg.value if arg.__class__ is self.__class__ else arg) for arg in args),
            **{k: (v.value if v.__class__ is self.__class__ else v) for k, v in kwargs.items()},
        ))

    def _arrays(self) -> Iterable[ArrayLike]:
        if numpy.ndim(self.value) == 0:
            return ()
        return (self.value,)

    def astype(self, dtype: DType, order: str = 'K', casting: str = 'unsafe',
               subok: bool = True, copy: bool = True) -> "Quantity":
        if numpy.ndim(self.value) == 0:
            if copy:
                return ValueExpr(self.value.copy())
            return self
        return ValueExpr(self.value.astype(dtype, order=order, casting=casting, subok=subok, copy=copy))

    def simplify(self) -> "PartialExpr":
        if numpy.count_nonzero(self.value) == 0:
            return ZeroExpr(self.shape)
        return self


@dataclass(frozen=True, init=False)
class FuncExpr(PartialExpr):
    func: PartialExpr = ValueExpr(lambda: 0)
    args: Tuple[PartialExpr, ...] = ()
    kwargs: Mapping[str, PartialExpr] = field(default_factory=lambda: {})

    def __init__(self, func: PartialExpr, args: Tuple[PartialExpr, ...], kwargs: Mapping[str, PartialExpr]):
        arrays = [a for params in ((func,), args, kwargs.values()) for param in params for a in param._arrays()]
        if len(arrays) > 1:
            shape = arrays[0].shape

            def _merge_dim(a: int, b: int) -> int:
                if a == 1:
                    return b
                if b == 1 or a == b:
                    return a
                raise ValueError("Mismatched array dimensions in PartialExpr")

            for array in arrays:
                if len(array.shape) != len(shape):
                    raise ValueError("Mismatched array dimensions in PartialExpr")
                shape = tuple(_merge_dim(a, b) for a, b in zip(array.shape, shape))

            func = numpy.broadcast_to(func, shape)
            args = tuple(numpy.broadcast_to(arg, shape) for arg in args)
            kwargs = {k: numpy.broadcast_to(v, shape) for k, v in kwargs.items()}

        object.__setattr__(self, "func", func)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "kwargs", kwargs)
        super().__init__(self._placeholder().shape)

    def _placeholder(self) -> Any:
        return self.func._placeholder()(
            *(arg._placeholder() for arg in self.args),
            **{k: v._placeholder() for k, v in self.kwargs.items()},
        )

    def locals(self) -> Dict[str, Any]:
        l = self.func.locals()
        for arg in itertools.chain(self.args, self.kwargs.values()):
            l.update(arg.locals())
        return l

    def params(self) -> List[str]:
        return list(dict.fromkeys(
            p
            for args in ((self.func,), self.args, self.kwargs.values())
            for arg in args
            for p in arg.params()
        ))

    def expr(self) -> str:
        return f"{self.func.expr()}({', '.join(arg.expr() for arg in self.args)}, {', '.join(f'{k}={v.expr()}' for k, v in self.kwargs.items())})"

    def __getitem__(self, key: Any) -> "PartialExpr":
        return FuncExpr(
            self.func[key],
            tuple(arg[key] for arg in self.args),
            {k: arg[key] for k, arg in self.kwargs.items()},
        )

    # TODO: handle reduction of dimensions by e.g. sum
    def _distribute_func(self, func: Callable, args: Sequence[Any], kwargs: Mapping[str, Any]) -> "PartialExpr":
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, PartialExpr):
                if not arg.__class__ is self.__class__ or len(arg.args) != len(self.args) or set(arg.kwargs.keys()) != set(self.kwargs.keys()):
                    raise TypeError(f"Inconsistent PartialExprs passed to {func.__name__}")
        return FuncExpr(
            func(
                *(arg.func if arg.__class__ is self.__class__ else arg for arg in args),
                **{k: v.func if v.__class__ is self.__class__ else v for k, v in kwargs.items()},
            ),
            tuple(
                func(
                    *(arg.args[i] if arg.__class__ is self.__class__ else arg for arg in args),
                    **{k: v.args[i] if v.__class__ is self.__class__ else v for k, v in kwargs.items()},
                )
                for i, _ in enumerate(self.args)
            ),
            {
                k: func(
                    *(arg.kwargs[k] if arg.__class__ is self.__class__ else arg for arg in args),
                    **{k: v.kwargs[k] if v.__class__ is self.__class__ else v for k, v in kwargs.items()},
                )
                for k in self.kwargs
            },
        )

    def _arrays(self) -> Iterable[ArrayLike]:
        for arg in itertools.chain((self.func,), self.args, self.kwargs.values()):
            yield from arg._arrays()

    def astype(self, dtype: DType, order: str = 'K', casting: str = 'unsafe',
               subok: bool = True, copy: bool = True) -> "Quantity":
        return FuncExpr(
            self.func.astype(dtype, order=order, casting=casting, subok=subok, copy=copy),
            tuple(
                arg.astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
                for arg in self.args
            ),
            {
                k: v.astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
                for k, v in self.kwargs.items()
            },
        )

    def simplify(self) -> "PartialExpr":
        args = tuple(a.simplify() for a in self.args)
        kwargs = {k: a.simplify() for k, a in self.kwargs.items()}
        if isinstance(self.func, ValueExpr):
            if self.func.value == numpy.add:
                if len(kwargs) == 0 and len(args) == 2:
                    if isinstance(args[0], ZeroExpr):
                        return args[1]
                    elif isinstance(args[1], ZeroExpr):
                        return args[0]
            elif self.func.value == numpy.subtract:
                if len(kwargs) == 0 and len(args) == 2:
                    if isinstance(args[0], ZeroExpr):
                        return -args[1]
                    elif isinstance(args[1], ZeroExpr):
                        return args[0]
            elif self.func.value == numpy.negative or self.func.value == numpy.positive:
                if len(kwargs) == 0 and len(args) == 1:
                    if isinstance(args[0], ZeroExpr):
                        return ZeroExpr()
            elif self.func.value == numpy.multiply:
                if len(kwargs) == 0 and len(args) == 2:
                    if isinstance(args[0], ZeroExpr) or isinstance(args[1], ZeroExpr):
                        return ZeroExpr()
                    elif isinstance(args[0], ValueExpr) and numpy.count_nonzero(args[0].value - 1) == 0:
                        return args[1]
                    elif isinstance(args[1], ValueExpr) and numpy.count_nonzero(args[1].value - 1) == 0:
                        return args[0]
            elif self.func.value == numpy.true_divide:
                if len(kwargs) == 0 and len(args) == 2:
                    if isinstance(args[0], ZeroExpr):
                        return ZeroExpr()
                    elif isinstance(args[1], ValueExpr) and numpy.count_nonzero(args[1].value - 1) == 0:
                        return args[0]
            elif self.func.value == numpy.power:
                if len(kwargs) == 0 and len(args) == 2:
                    if isinstance(args[0], ZeroExpr):
                        return ZeroExpr()
                    elif isinstance(args[1], ZeroExpr):
                        return ValueExpr(1)
            elif self.func.value == numpy.exp:
                if len(kwargs) == 0 and len(args) == 1:
                    if isinstance(args[0], ZeroExpr):
                        return ValueExpr(1)
            elif self.func.value == numpy.sin:
                if len(kwargs) == 0 and len(args) == 1:
                    if isinstance(args[0], ZeroExpr):
                        return ZeroExpr()
            elif self.func.value == numpy.cos:
                if len(kwargs) == 0 and len(args) == 1:
                    if isinstance(args[0], ZeroExpr):
                        return ValueExpr(1)
        return FuncExpr(self.func, args, kwargs)


@dataclass(frozen=True, init=False)
class UFuncExpr(FuncExpr):
    method: str = "exp"

    def __init__(self, func: PartialExpr, args: Tuple[PartialExpr, ...], kwargs: Mapping[str, PartialExpr], method: str):
        object.__setattr__(self, "method", method)
        super().__init__(func, args, kwargs)

    def _placeholder(self) -> Any:
        return getattr(self.func._placeholder(), self.method)(
            *(arg._placeholder() for arg in self.args),
            **{k: v._placeholder() for k, v in self.kwargs.items()},
        )

    def expr(self) -> str:
        if self.method == '__call__':
            return f"{self.func.expr()}({', '.join(arg.expr() for arg in self.args)}, {', '.join(f'{k}={v.expr()}' for k, v in self.kwargs.items())})"
        return f"{self.func.expr()}.{self.method}({', '.join(arg.expr() for arg in self.args)}, {', '.join(f'{k}={v.expr()}' for k, v in self.kwargs.items())})"

    def __getitem__(self, key: Any) -> "PartialExpr":
        return UFuncExpr(
            self.func[key],
            tuple(arg[key] for arg in self.args),
            {k: arg[key] for k, arg in self.kwargs.items()},
            self.method,
        )

    def _distribute_func(self, func: Callable, args: Sequence[Any], kwargs: Mapping[str, Any]) -> "PartialExpr":
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, PartialExpr):
                if not arg.__class__ is self.__class__ or len(arg.args) != len(self.args) or set(arg.kwargs.keys()) != set(self.kwargs.keys()):
                    raise TypeError(f"Inconsistent PartialExprs passed to {func.__name__}")
        return UFuncExpr(
            func(
                *(arg.func if arg.__class__ is self.__class__ else arg for arg in args),
                **{k: v.func if v.__class__ is self.__class__ else v for k, v in kwargs.items()},
            ),
            tuple(
                func(
                    *(arg.args[i] if arg.__class__ is self.__class__ else arg for arg in args),
                    **{k: v.args[i] if v.__class__ is self.__class__ else v for k, v in kwargs.items()},
                )
                for i, _ in enumerate(self.args)
                ),
            {
                k: func(
                    *(arg.kwargs[k] if arg.__class__ is self.__class__ else arg for arg in args),
                    **{k: v.kwargs[k] if v.__class__ is self.__class__ else v for k, v in kwargs.items()},
                )
                for k in self.kwargs
            },
            self.method,
        )

    def astype(self, dtype: DType, order: str = 'K', casting: str = 'unsafe',
               subok: bool = True, copy: bool = True) -> "Quantity":
        return UFuncExpr(
            self.func.astype(dtype, order=order, casting=casting, subok=subok, copy=copy),
            tuple(
                arg.astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
                for arg in self.args
            ),
            {
                k: v.astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
                for k, v in self.kwargs.items()
            },
            self.method,
        )

    def simplify(self) -> "PartialExpr":
        if self.method == "__call__":
            simplified = super().simplify()
            if isinstance(simplified, FuncExpr):
                return UFuncExpr(simplified.func, simplified.args, simplified.kwargs, self.method)
            else:
                return simplified
        return self


def get_var(var: Union[str, Callable], data: Dataset) -> DataArray:
    if callable(var):
        sig = describe(var)
        return var(*(data[v] for v in sig))
    return data[var]


def fit_jack(data: Union[Dataset, Tuple[Dataset, ...]], var: Union[str, Callable, Tuple[Union[str, Callable], ...]], model: Union[IntoModel, Tuple[IntoModel, ...]],
             cost: Type[Cost] = NDCorrelatedChiSquared, dim: str = 'jack', keep: Sequence[str] = (), covariant_dims: Sequence[str] = (),
             ncall: Optional[int] = None, tol: Optional[float] = None, **kwargs) -> Dataset:
    if not isinstance(var, tuple):
        var = (var,)
    if not isinstance(model, tuple):
        model = (model,)
    if len(var) != len(model):
         raise ValueError("model and var should be the same length")
    if not isinstance(data, tuple):
        data = (data,) * len(var)
    if len(data) != len(var):
        raise ValueError("data and var should be the same length")

    model = tuple(Model.new(m) for m in model)
    data_var = tuple(get_var(v, d) for v, d in zip(var, data))

    for d, v, m in zip(data, data_var, model):
        units, unwrapped, limits = check_model_units(d, v, m, kwargs)

    if isinstance(keep, str):
        keep = (keep,)

    partial_model = tuple(
        Model.new(m)(**{
            k: d.get(k, Placeholder(k) * units.get(k, one))
            for k in describe(m)
        }) * one
        for d, m in zip(data, model)
    )

    for v, m in zip(data_var, partial_model):
        m_unwrap = unwrap_xarray(m)
        v_unwrap = unwrap_xarray(v)
        if v_unwrap.dimension != m_unwrap.dimension:
            raise ValueError(f"Model and data have incompatible mass dimensions: {m_unwrap.dimension} and {v_unwrap.dimension}")
        if v_unwrap.dimension != Scalar and v_unwrap.scale != m_unwrap.scale:
            raise ValueError("Mismatched scale for model and data")

    costs = tuple(
        cost(
            v, m,
            value=functools.partial(value_jack, dim=dim),
            covariance=functools.partial(covariance_jack, dim=dim),
            covariant_dims=covariant_dims,
        )
        for v, m in zip(data_var, partial_model)
    )

    minuit = Minuit(sum(c for c in costs), **unwrapped)
    minuit.tol = tol
    for k, v in limits.items():
        minuit.limits[k] = v

    if len(keep) == 0:
        minuit.migrad(ncall=ncall)
        # print(f"{tuple(v for v in m.values)} -> {sum(c.__call__(tuple(v for v in m.values)) for c in costs)}")
        chi2 = minuit.fval
        dof = sum(reduce(mul, (v for k, v in d.dims.items() if k != dim)) for d in data) - len(units)
        # TODO: consider preserving attrs
        return Dataset(
            {
                k: minuit.values[k] * units[k] for k in unwrapped.keys()
            },
            attrs={
                "chi2": chi2,
                "dof": dof,
            }
        )
    else:
        first_data = data[0]

        def indices(dims: Tuple[str, ...]) -> Iterator[Dict[str, int]]:
            if len(dims) == 0:
                yield {}
            else:
                for index in indices(dims[1:]):
                    for i in first_data[dims[0]]:
                        index[dims[0]] = i
                        yield index

        def fit_at(index: Mapping[str, int]) -> Tuple[Dataset, float]:
            dim_idx = index.get(dim, 0)
            jack_index: Mapping[Hashable, int] = {
                k: v
                for k, v in index.items()
                if k != dim
            }
            for c in costs:
                c.set_function(
                    lambda v: v.sel(jack_index),
                    lambda var: var.sel({dim: dim_idx}),
                    lambda a, b: covariance_jack(a, b, dim),
                )
            minuit.migrad(ncall=ncall)
            # print(f"{tuple(v for v in m.values)} -> {sum(c.__call__(tuple(v for v in m.values)) for c in costs)}")
            return (
                Dataset(
                    {
                        k: minuit.values[k] * units[k] for k in unwrapped.keys()
                    },
                ),
                minuit.fval or 0.0,
            )

        index_iter = iter(indices(tuple(keep)))

        idx = next(index_iter)
        dof = sum(reduce(mul, (v for k, v in d[idx].dims.items() if k != dim)) for d in data) - len(units)
        result, chi2 = fit_at(idx)
        expand: Mapping[Hashable, int] = {
            dim: len(first_data[dim])
            for dim in keep
        }
        result = result.expand_dims(expand)
        for k in unwrapped.keys():
            result[k] = result[k].copy()
        chi2 = DataArray(chi2).expand_dims(expand).copy()
        for idx in index_iter:
            result_idx, chi2[idx] = fit_at(idx)
            for k in unwrapped.keys():
                result[k][idx] = result_idx[k].variable

        result.attrs = {
            'chi2': chi2,
            'dof': dof,
        }

        return result
