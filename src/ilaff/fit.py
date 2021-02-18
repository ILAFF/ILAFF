from abc import ABC, abstractmethod
from functools import partial
from iminuit import Minuit, describe
import iminuit.cost
import resample
import functools
import numpy
from xarray import Dataset, DataArray, Variable, broadcast
from typing import Callable, Union, Mapping, Sequence, Any, Tuple, Type, Optional
from inspect import signature
import abc
import itertools

from ilaff.units import Quantity, QuantityIndex, Scalar

ModelFn = Callable[..., Quantity]


def jackknife(data: Dataset, dim: str = 'configuration', jackdim: str = 'jack', fn: Callable[[Dataset, str], Dataset] = lambda v: numpy.mean(v, axis=0)) -> Dataset:
    data = data.transpose(dim, *(d for d in data.dims if d != dim))

    jack = {
        k: (#Variable(
            [jackdim] + [d for d in v.dims if d != dim],
            numpy.stack([fn(v.data)] + [fn(a) for a in resample.jackknife.resample(v.data)]),
            #v.attrs,
            #v.encoding,
        ) if dim in v.dims else v
        for k, v in data.variables.items()
    }

    # TODO: figure out how to use variables directly here
    d = Dataset(
        {k: jack[k] for k in data},
        coords={k: jack[k] for k in data.coords},
        attrs=data.attrs,
    )

    return d


class Cost(iminuit.cost.Cost, abc.ABC):
    @abc.abstractmethod
    def __init__(self, data: Dataset, var: Union[str, Callable[..., Quantity]], model: ModelFn, value: Callable[[DataArray], DataArray], covariance: Callable[[DataArray, DataArray], DataArray], verbose: int = 0): ...

    @abc.abstractmethod
    def set_data(self, data: Dataset) -> None: ...


def _unwrap_quantity(quantity: Optional[Union[Quantity, DataArray]]) -> Any:
    if quantity is None:
        return None
    if isinstance(quantity, Quantity):
        return quantity.value
    if isinstance(quantity.variable._data, Quantity):
        quantity = quantity.copy()
        quantity.data = quantity.variable._data.value
    elif isinstance(getattr(quantity.variable._data, "array", None), QuantityIndex):
        quantity = DataArray(
            quantity.variable._data.array.array,
            coords=quantity.coords,
            dims=quantity.dims,
            name=quantity.name,
            attrs=quantity.attrs,
            indexes=quantity.indexes,
        )
    return quantity


class NDCorrelatedChiSquared(Cost):
    def __init__(self, data: Dataset, var: Union[str, Callable], model: ModelFn, dim: str, value: Callable[[DataArray], DataArray], covariance: Callable[[DataArray, DataArray], DataArray], verbose: int = 0):
        self.var = var
        self.model = model
        self.value = value
        self.covariance = covariance
        self.args = describe(self.model)
        self.bound_args = [arg for arg in self.args if arg in data]
        self.dim = dim
        self.set_data(data)

        iminuit.cost.Cost.__init__(self, [arg for arg in self.args if arg not in data], verbose)

    def set_data(self, data: Dataset) -> None:
        if callable(self.var):
            sig = describe(self.var)
            y = _unwrap_quantity(self.var(*(data[v] for v in sig)))
        else:
            y = _unwrap_quantity(data[self.var])
        args = [data.get(v) for v in self.args]
        broadcasted = iter(broadcast(y, *(_unwrap_quantity(v) for v in args if v is not None)))
        self.y = next(broadcasted).data
        self.data_args = [
            next(broadcasted).data if v is not None else None for v in args
        ]
        if self.dim is not None:
            self.axis = y.dims.index(self.dim)
        else:
            self.axis = None

    def wrapped_model(self, *args):
        args = iter(args)
        return self.model(*(
            data_arg if data_arg is not None else next(args)
            for data_arg in self.data_args
        ))

    def _call(self, args) -> float:
        y = self.y
        ym = self.wrapped_model(*args)
        r = y - ym
        # TODO incorporate correlations along dimensions
        return (self.value(r, self.axis)**2 / self.covariance(r, r, self.axis)).sum()


def unwrap_xarray(a: Any) -> Any:
    variable = getattr(a, 'variable', None)
    if variable is None:
        return a
    data = variable._data
    if getattr(data, "shape", None) == ():
        return data[()]
    return getattr(data, "array", data)


def check_model_units(data: Dataset, var: Union[str, Callable], model: ModelFn, kwargs: Mapping[str, Any]) -> Tuple[Mapping[str, Quantity], Mapping[str, Any]]:
    data = data.isel({dim: 0 for dim in data.dims})
    if callable(var):
        sig = describe(var)
        y = unwrap_xarray(var(*(data[v] for v in sig)))
    else:
        y = unwrap_xarray(data[var])

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
        None,
    )

    sig = describe(model)

    if scale is None:
        return {k: 1 for k in sig}, kwargs

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

    def check_unwrap(k: str, v: Quantity) -> Any:
        q: Optional[Quantity] = None
        for prefix in ("", "error_", "limit_", "fix_"):
            if k.startswith(prefix) and k[len(prefix):] in params:
                q = params[k]
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
                q = params[k]
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
    ym = model(*args)
    if isinstance(ym, Quantity):
        if y.dimension != ym.dimension:
            raise ValueError(f"Model and data have incompatible mass dimensions: {y.dimension} and {ym.dimension}")
        if y.dimension != Scalar and y.scale != ym.scale:
            raise ValueError("Mismatched scale for model and data")
        return {k: Quantity(1, v.dimension, v.scale) for k, v in params.items()}, unwrapped
    else:
        raise NotImplementedError("Paramater unit inference not yet supported")


def value_jack(v: DataArray, dim: str = 'jack') -> DataArray:
    return v.isel({dim: 0})


def error_jack(v: DataArray, dim: str = 'jack') -> DataArray:
    vbar = v.isel({dim: slice(1, None)})
    N = len(vbar[dim])
    return (((vbar - vbar.mean(dim))**2).sum(dim) * (N - 1) / N)**0.5


def fit_jack(data: Dataset, var: Union[str, Callable, Tuple[Union[str, Callable], ...]], model: Union[ModelFn, Tuple[ModelFn]],
             cost: Type[Cost] = NDCorrelatedChiSquared, dim: str = 'jack', keep: Sequence[str] = (), **kwargs) -> Dataset:
    if isinstance(var, tuple):
        if not isinstance(model, tuple) or len(var) != len(model):
            raise ValueError("model and var should be the same length")
        for v, m in zip(var, model):
            units, unwrapped = check_model_units(data, v, m, kwargs)
    else:
        if isinstance(model, tuple):
            raise ValueError("model and var should be the same length")
        units, unwrapped = check_model_units(data, var, model, kwargs)
    if len(keep) == 0:
        if isinstance(var, tuple):
            c = sum(
                cost(
                    data, v, m, dim,
                    value=lambda v, axis: v[(slice(None),) * axis + (0,)],
                    covariance=lambda a, b, axis: (
                        (a[(slice(None),) * axis + (slice(1, None),)] - a[(slice(None),) * axis + (slice(1, None),)].mean(axis=axis, keepdims=True))
                        * (b[(slice(None),) * axis + (slice(1, None),)] - b[(slice(None),) * axis + (slice(1, None),)].mean(axis=axis, keepdims=True))
                    ).sum(axis=axis) * (a.shape[axis] - 2) / (a.shape[axis] - 1)
                )
                for v, m in zip(var, model)
            )
        else:
            c = cost(
                data, var, model, dim,
                value=lambda v, axis: v[(slice(None),) * axis + (0,)],
                covariance=lambda a, b, axis: (
                    (a[(slice(None),) * axis + (slice(1, None),)] - a[(slice(None),) * axis + (slice(1, None),)].mean(axis=axis, keepdims=True))
                    * (b[(slice(None),) * axis + (slice(1, None),)] - b[(slice(None),) * axis + (slice(1, None),)].mean(axis=axis, keepdims=True))
                ).sum(axis=axis) * (a.shape[axis] - 2) / (a.shape[axis] - 1)
            )
        m = Minuit(c, **unwrapped)
        m.migrad()
        for k in unwrapped.keys():
            print(f"{k}: {m.values[k]} +- {m.errors[k]}")
        print(m.covariance)
        return Dataset(
            {
                k: m.values[k] * units[k] for k in unwrapped.keys()
            },
            attrs=data.attrs,
        )
    else:
        raise NotImplementedError("kept dimensions not yet supported")
        # def indices(dims: Tuple[str]) -> Iterator[Mapping[str, int]]:
        #     if len(dims) == 0:
        #         yield {}
        #     else:
        #         index = indices(dims[1:])
        #         for i in ??:
        #             index[dims[0]] = i
        #             yield index

        # c = cost(
        #     data.isel(index), var, model,
        #     value=lambda v: v.isel({dim: 0}),
        #     covariance=lambda a, b: (
        #         (a.isel({dim: slice(1, None)}) - a.isel({dim: slice(1, None)}).mean(dim))
        #         * (b.isel({dim: slice(1, None)}) - b.isel({dim: slice(1, None)}).mean(dim))
        #     ).sum(dim) * (len(a[dim]) - 2) / (len(a[dim]) - 1)
        # )
        # m = Minuit(c, **kwargs)
        # m.migrad()
        # res = Dataset({
        #     k: (keep, 
        # }, coords={
        # })
        # for index in indices(keep):
        #     m.migrad()
        # return Dataset(
        #     {
        #         k: v for k,v in m.values.items()
        #     },
        # )
