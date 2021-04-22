from abc import ABC, abstractmethod
from functools import partial
from iminuit import Minuit, describe
import iminuit.cost
import resample
import functools
import numpy
from xarray import Dataset, DataArray, Variable, broadcast
from typing import Callable, Union, Mapping, Sequence, Any, Tuple, Type, Optional, Iterator
from inspect import signature, Signature
import abc
import itertools
from dataclasses import dataclass, field
from functools import reduce
from operator import mul

from ilaff.units import Quantity, QuantityIndex, Scalar, one
from ilaff.units.quantity import _upcast_types


IntoModel = Union["Model", Callable[..., Quantity]]


class Model(ABC):
    @staticmethod
    def new(fn: IntoModel) -> "Model":
        if isinstance(fn, Model):
            return fn
        if callable(fn):
            return ModelFn(fn)
        raise TypeError(f"'{fn!r}' cannot be used as a model")

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Quantity:
        ...

    def evaluate(self, data: Dataset) -> DataArray:
        return self(**{key: data[key] for key in signature(self).parameters.keys()})

    def partial(self, *data: Dataset, **kwargs: Union[DataArray, Quantity]) -> "Model":
        data_args = {
            key: next(iter(d[key] for d in data if key in d))
            for key in signature(self).parameters.keys()
            if key not in kwargs and any(key in d for d in data)
        }
        return PartialModel(self, {**kwargs, **data_args})

    def __neg__(self) -> "Model":
        return -1 * self

    def __pos__(self) -> "Model":
        return self

    def __add__(self, other: IntoModel) -> "Model":
        return AddModel(self, Model.new(other))

    def __radd__(self, other: IntoModel) -> "Model":
        return AddModel(Model.new(other), self)

    def __sub__(self, other: IntoModel) -> "Model":
        return self + (-Model.new(other))

    def __rsub__(self, other: IntoModel) -> "Model":
        return Model.new(other) + (-self)

    def __mul__(self, other: Union[Quantity, float]) -> "Model":
        return ScaleModel(other, self)

    def __rmul__(self, other: Union[Quantity, float]) -> "Model":
        return ScaleModel(other, self)

    def __truediv__(self, other: Union[Quantity, float]) -> "Model":
        return ScaleModel(1 / other, self)


@dataclass(frozen=True)
class ModelFn(Model):
    __wrapped__: Callable[..., Quantity]

    def __call__(self, *args: Any, **kwargs: Any) -> Quantity:
        return self.__wrapped__(*args, **kwargs)


@dataclass(frozen=True)
class PartialModel(Model):
    __wrapped__: Model
    kwargs: Mapping[str, Any]
    __signature__: Signature = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        sig = signature(self.__wrapped__)
        sig = sig.replace(parameters=(
            v
            for k, v in sig.parameters.items()
            if k not in self.kwargs
        ))
        object.__setattr__(self, '__signature__', sig)

    def __call__(self, *args: Any, **kwargs: Any) -> Quantity:
        bound = self.__signature__.bind(*args, **kwargs)
        return self.__wrapped__(**bound.arguments, **self.kwargs)


@dataclass(frozen=True)
class ScaleModel(Model):
    scale: Union[Quantity, float]
    __wrapped__: Model

    def __call__(self, *args: Any, **kwargs: Any) -> Quantity:
        return self.scale * self.__wrapped__(*args, **kwargs)


@dataclass(frozen=True)
class AddModel(Model):
    left: Model
    right: Model
    __signature__: Signature = field(init=False, repr=False, compare=False)
    _left_signature: Signature = field(init=False, repr=False, compare=False)
    _right_signature: Signature = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        left_sig = signature(self.left)
        right_sig = signature(self.right)
        sig = left_sig.replace(parameters=itertools.chain(
            (
                v
                for k, v in left_sig.parameters.items()
            ),
            (
                v
                for k, v in right_sig.parameters.items()
                if k not in left_sig.parameters
            ),
        ))
        object.__setattr__(self, '__signature__', sig)
        object.__setattr__(self, '_left_signature', left_sig)
        object.__setattr__(self, '_right_signature', right_sig)

    def __call__(self, *args: Any, **kwargs: Any) -> Quantity:
        bound = self.__signature__.bind(*args, **kwargs)
        return (
            self.left(**{k: bound.arguments[k] for k in self._left_signature.parameters if k in bound.arguments})
            + self.right(**{k: bound.arguments[k] for k in self._right_signature.parameters if k in bound.arguments})
        )


_upcast_types += [ModelFn, PartialModel, ScaleModel, AddModel]


def evaluate(model: IntoModel, data: Dataset) -> DataArray:
    return Model.new(model).evaluate(data)


def partial(model: IntoModel, *data: Dataset, **kwargs: Union[DataArray, Quantity]) -> Model:
    return Model.new(model).partial(*data, **kwargs)


def jackknife(data: Union[Dataset, DataArray], dim: str = 'configuration', jackdim: str = 'jack', fn: Callable[[Dataset, str], Dataset] = lambda v: numpy.mean(v, axis=0), bin_width: int = 1) -> Dataset:
    data = data.transpose(dim, *(d for d in data.dims if d != dim))
    if bin_width < 1:
        raise ValueError("Bin width must be at least one")
    if len(data[dim]) < bin_width * 2:
        raise ValueError("Insufficient configurations for jackknife")

    if isinstance(data, Dataset):
        # TODO: consider if bin should always average
        jack = {
            k: (#Variable(
                [jackdim] + [d for d in v.dims if d != dim],
                numpy.stack(
                    [fn(v.data)]
                    + [
                        fn(a)
                        for a in resample.jackknife.resample(
                            v.data if bin_width == 1 else
                            numpy.add.reduceat(
                                v.data,
                                range(0, v.data.shape[0], bin_width),
                            )[:(-1 if v.data.shape[0] % bin_width != 0 else None)] / bin_width,
                            copy=False,
                        )
                    ]
                ),
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
    else:
        return DataArray(
            data=numpy.stack(
                [fn(data.data)]
                + [
                    fn(a)
                    for a in resample.jackknife.resample(
                        data.data if bin_width == 1 else
                        numpy.add.reduceat(
                            data.data,
                            range(0, data.data.shape[0], bin_width),
                        )[:(-1 if data.data.shape[0] % bin_width != 0 else None)] / bin_width,
                        copy=False,
                    )
                ]
            ),
            coords=data.coords,
            dims=[jackdim] + [d for d in data.dims if d != dim],
            name=data.name,
            attrs=data.attrs,
        )



class Cost(iminuit.cost.Cost, abc.ABC):
    @abstractmethod
    def __init__(self, data: Dataset, var: Union[str, Callable[..., Quantity]], model: Model, value: Callable[[DataArray], DataArray], covariance: Callable[[DataArray, DataArray], DataArray], covariant_dims: Sequence[str] = (), verbose: int = 0): ...

    @abstractmethod
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
    def __init__(self, data: Dataset, var: Union[str, Callable], model: Model, units: Mapping[str, Quantity], dim: str, value: Callable[[DataArray], DataArray], covariance: Callable[[DataArray, DataArray], DataArray], covariant_dims: Sequence[str] = (), verbose: int = 0):
        self.var = var
        self.model = model
        self.value = value
        self.covariance = covariance
        self.args = describe(self.model)
        self.bound_args = [arg for arg in self.args if arg in data]
        self.dim = dim
        self.units = units
        self.covariant_dims = covariant_dims
        self.set_data(data)

        iminuit.cost.Cost.__init__(self, [arg for arg in self.args if arg not in data], verbose)

    def set_data(self, data: Dataset) -> None:
        if callable(self.var):
            sig = describe(self.var)
            y = self.var(*(data[v] for v in sig))
        else:
            y = data[self.var]
        args = [data.get(v) for v in self.args]
        # TODO: check that data args match units
        # broadcasted = iter(broadcast(y, *(v for v in args if v is not None)))
        # self.y = next(broadcasted).data
        # self.data_args = [
        #     next(broadcasted).data if v is not None else None for v in args
        # ]
        self.y, self.data_args = y, args
        # if self.dim is not None:
        #     self.axis = y.dims.index(self.dim)
        # else:
        #     self.axis = None

    def wrapped_model(self, *args):
        args = iter(args)
        return self.model(*(
            data_arg if data_arg is not None else next(args) * self.units.get(kw, 1)
            for (kw, data_arg) in zip(self.args, self.data_args)
        ))

    def _call(self, args) -> float:
        y = self.y
        ym = self.wrapped_model(*args)
        r = y - ym
        # covariant_dims = r.attrs.get("covariant_dims", ())
        covariant_dims = self.covariant_dims
        if len(covariant_dims) == 0:
            if self.dim is not None:
                axis = r.dims.index(self.dim)
            else:
                axis = None
            r = r.data
            chi2 = (self.value(r, axis)**2 / self.covariance(r, r, axis)).sum()
            try:
                return chi2.in_unit(one)
            except AttributeError:
                return chi2
        else:
            #print(f"covariant: {covariant_dims}")
            #print(args)
            #print(y)
            #print(ym)
            #print(r)
            r = r.stack(covariant_dims=covariant_dims)
            if self.dim is not None:
                axis = r.dims.index(self.dim)
            else:
                axis = None
            rbar = r.expand_dims('covariant_dims_2', -2).data
            r = r.expand_dims('covariant_dims_2', -1).data
            cov = self.covariance(r, rbar, axis)
            try:
                inv_cov = numpy.linalg.inv(cov)
            except e:
                print(args)
                raise e
            chi2 = (self.value(rbar, axis) @ inv_cov @ self.value(r, axis)).sum()
            try:
                return chi2.in_unit(one)
            except AttributeError:
                return chi2


def unwrap_xarray(a: Any) -> Any:
    variable = getattr(a, 'variable', None)
    if variable is None:
        return a
    data = variable._data
    if getattr(data, "shape", None) == ():
        return data[()]
    return getattr(data, "array", data)


def check_model_units(data: Dataset, var: Union[str, Callable], model: IntoModel, kwargs: Mapping[str, Any]) -> Tuple[Mapping[str, Quantity], Mapping[str, Any], Mapping[str, Any]]:
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
    ym = unwrap_xarray(model(*args))
    if not isinstance(ym, Quantity):
        ym = ym * one
    if y.dimension != ym.dimension:
        raise ValueError(f"Model and data have incompatible mass dimensions: {ym.dimension} and {y.dimension}")
    if y.dimension != Scalar and y.scale != ym.scale:
        raise ValueError("Mismatched scale for model and data")
    return (
        {k: Quantity(1, v.dimension, v.scale) for k, v in params.items()},
        {k: v for k, v in unwrapped.items() if k in params},
        {k[6:]: v for k, v in unwrapped.items() if k[:6] == "limit_" and k[6:] in params},
    )


def value_jack(v: DataArray, dim: str = 'jack') -> DataArray:
    return v.isel({dim: 0})


def error_jack(v: DataArray, dim: str = 'jack') -> DataArray:
    vbar = v.isel({dim: slice(1, None)})
    N = len(vbar[dim])
    return (((vbar - vbar.mean(dim))**2).sum(dim) * (N - 1) / N)**0.5


def covariance_jack(v: DataArray, w: DataArray, dim: str = 'jack') -> DataArray:
    vbar = v.isel({dim: slice(1, None)})
    wbar = w.isel({dim: slice(1, None)})
    N = len(vbar[dim])
    return ((vbar - vbar.mean(dim)) * (wbar - wbar.mean(dim))).sum(dim) * (N - 1) / N


def fit_jack(data: Union[Dataset, Tuple[Dataset, ...]], var: Union[str, Callable, Tuple[Union[str, Callable], ...]], model: Union[IntoModel, Tuple[IntoModel]],
             cost: Type[Cost] = NDCorrelatedChiSquared, dim: str = 'jack', keep: Sequence[str] = (), covariant_dims: Sequence[str] = (), **kwargs) -> Dataset:
    if isinstance(var, tuple):
        if not isinstance(model, tuple) or len(var) != len(model):
            raise ValueError("model and var should be the same length")
        model = tuple(Model.new(m) for m in model)
        if not isinstance(data, tuple):
            data = (data,) * len(var)
        for d, v, m in zip(data, var, model):
            units, unwrapped, limits = check_model_units(d, v, m, kwargs)
    else:
        if isinstance(model, tuple):
            raise ValueError("model and var should be the same length")
        if isinstance(data, tuple):
            raise ValueError("data and var should be the same length")
        model = Model.new(model)
        units, unwrapped, limits = check_model_units(data, var, model, kwargs)

    if isinstance(keep, str):
        keep = (keep,)

    if dim in keep:
        new_dim = f"_{dim}"
        if isinstance(data, tuple):
            while any(new_dim in d.dims for d in data):
                new_dim = f"_{new_dim}"
            expanded_data = tuple(Dataset(
                data_vars={
                    k: v.rename({dim: new_dim}).expand_dims({dim: len(d[dim])}).copy() if dim in v.dims else v
                    for k, v in d.data_vars.items()
                },
                coords={
                    k: v.rename({dim: new_dim}).expand_dims({dim: len(d[dim])}).copy() if dim in v.dims else v
                    for k, v in d.coords.items()
                },
                attrs=d.attrs,
            ) for d in data)
            for i, d in enumerate(data):
                for key in itertools.chain(d.data_vars, d.coords):
                    if dim in d[key].dims:
                        expanded_data[i][key][{new_dim: 0}] = d[key].variable
        else:
            while new_dim in data.dims:
                new_dim = f"_{new_dim}"
            expanded_data = Dataset(
                data_vars={
                    k: v.rename({dim: new_dim}).expand_dims({dim: len(data[dim])}).copy() if dim in v.dims else v
                    for k, v in data.data_vars.items()
                },
                coords={
                    k: v.rename({dim: new_dim}).expand_dims({dim: len(data[dim])}).copy() if dim in v.dims else v
                    for k, v in data.coords.items()
                },
                attrs=data.attrs,
            )
            for key in itertools.chain(data.data_vars, data.coords):
                if dim in data[key].dims:
                    expanded_data[key][{new_dim: 0}] = data[key].variable
        data = expanded_data
        dim = new_dim

    if isinstance(var, tuple):
        costs = [
            cost(
                d, v, m, units, dim,
                value=lambda v, axis: v[(slice(None),) * axis + (0,)],
                covariance=lambda a, b, axis: (
                    (a[(slice(None),) * axis + (slice(1, None),)] - a[(slice(None),) * axis + (slice(1, None),)].mean(axis=axis, keepdims=True))
                    * (b[(slice(None),) * axis + (slice(1, None),)] - b[(slice(None),) * axis + (slice(1, None),)].mean(axis=axis, keepdims=True))
                ).sum(axis=axis) * (a.shape[axis] - 2) / (a.shape[axis] - 1),
                covariant_dims=covariant_dims,
            )
            for d, v, m in zip(data, var, model)
        ]
        c = sum(costs)

        def set_data(data: Tuple[Dataset, ...]):
            for d, c in zip(data, costs):
                c.set_data(d)
    else:
        c = cost(
            data, var, model, units, dim,
            value=lambda v, axis: v[(slice(None),) * axis + (0,)],
            covariance=lambda a, b, axis: (
                (a[(slice(None),) * axis + (slice(1, None),)] - a[(slice(None),) * axis + (slice(1, None),)].mean(axis=axis, keepdims=True))
                * (b[(slice(None),) * axis + (slice(1, None),)] - b[(slice(None),) * axis + (slice(1, None),)].mean(axis=axis, keepdims=True))
            ).sum(axis=axis) * (a.shape[axis] - 2) / (a.shape[axis] - 1),
            covariant_dims=covariant_dims,
        )

        def set_data(data: Dataset):
            c.set_data(data)

    m = Minuit(c, **unwrapped)
    for k, v in limits.items():
        m.limits[k] = v

    if len(keep) == 0:
        m.migrad()
        chi2 = m.fval
        if isinstance(data, tuple):
            dof = sum(reduce(mul, (v for k, v in d.dims.items() if k != dim)) for d in data) - len(units)
            # TODO: consider preserving attrs
            attrs = {}
        else:
            dof = reduce(mul, (v for k, v in data.dims.items() if k != dim)) - len(units)
            attrs = data.attrs
        return Dataset(
            {
                k: m.values[k] * units[k] for k in unwrapped.keys()
            },
            attrs={
                "chi2": chi2,
                "dof": dof,
                **attrs,
            }
        )
    else:
        # TODO: check for mismatched dimensions
        if isinstance(data, tuple):
            first_data = data[0]
        else:
            first_data = data
        def indices(dims: Tuple[str]) -> Iterator[Mapping[str, int]]:
            if len(dims) == 0:
                yield {}
            else:
                for index in indices(dims[1:]):
                    for i in first_data[dims[0]]:
                        index[dims[0]] = i
                        yield index

        def fit_at(index: Mapping[str, int]) -> Tuple[DataArray, float]:
            if isinstance(data, tuple):
                set_data(tuple(d[index] for d in data))
            else:
                set_data(data[index])
            m.migrad()
            return (
                Dataset(
                    {
                        k: m.values[k] * units[k] for k in unwrapped.keys()
                    },
                ),
                m.fval,
            )

        index_iter = iter(indices(keep))

        idx = next(index_iter)
        if isinstance(data, tuple):
            dof = sum(reduce(mul, (v for k, v in d[idx].dims.items() if k != dim)) for d in data) - len(units)
            attrs = {}
        else:
            dof = reduce(mul, (v for k, v in data[idx].dims.items() if k != dim)) - len(units)
            attrs = data.attrs
        result, chi2 = fit_at(idx)
        result = result.expand_dims({
            dim: len(first_data[dim])
            for dim in keep
        })
        for k in unwrapped.keys():
            result[k] = result[k].copy()
        chi2 = DataArray(chi2).expand_dims({
            dim: len(first_data[dim])
            for dim in keep
        }).copy()
        for idx in index_iter:
            result_idx, chi2[idx] = fit_at(idx)
            for k in unwrapped.keys():
                result[k][idx] = result_idx[k].variable

        result.attrs = {
            'chi2': chi2,
            'dof': dof,
            **attrs
        }

        return result
