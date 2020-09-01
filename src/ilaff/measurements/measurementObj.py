from .measurementJack import measurementJack # type: ignore
from ilaff.units.quantity import Quantity

from dataclasses import dataclass
from typing import Any

import numpy as np # type: ignore

import resample as res # type: ignore


@dataclass(frozen=True, eq=False, order=False)
class measurement:
    iValue: Quantity
    dValue: Quantity
    
    def measResample(self, resampleType: str ) -> measurementJack:
        """
        Does resampling of type specified by resampleType.
        Returns new object
        measurementJack assumes jackknife only currently - fix?
        """
        if resampleType == 'jack':
            #TODO: Check for presence of resample package, use it if it exists, otherwise use our own implementation
            resFunc = res.jackknife.resample
            #resFunc = res.jackknife.jackknife
        #elif resampleType == 'boot':
        #    resFunc = bootstrap
        else:
            #TODO: Look at the other resampling methods in resample
            exit( 'invalid resampling method selected')
        #Now actually doing it    
        #TODO: make the unwrapping work better? Should be able to do this without the Quantity
        #resampled =  Quantity( resFunc(self.dValue.value)[:,0,:], self.dValue.dimension, self.dValue.scale)   #not entirely sure what the middle index is... Ask?

        #resampled = np.array( resFunc( np.ones_like, self.dValue.value ) )

        jackknifeMeans = np.concatenate([np.mean(sample, axis=0, keepdims=True) for sample in res.jackknife.resample(self.dValue.value)])
        jackQuant = Quantity( jackknifeMeans, self.dValue.dimension, self.dValue.scale)

        mean = np.mean( self.dValue, axis=0 )
        return measurementJack( self.iValue, mean, jackQuant )
        #return measurementJack( self.iValue, mean, mean )
        


        #=,>,<,>=,<= comparators
    def __eq__(self, other: Any) -> Any:
        if isinstance(other, measurement):            
            equal = False
            if self.iValue == other.iValue:
                if self.dValue == other.dValue:
                    equal = True
            return equal
        else:
            return False
    def __lt__(self, other: "measurement")-> Any:
        equal = False
        if self.iValue == other.iValue:
            if self.dValue < other.dValue:
                equal = True
        return equal
    def __le__(self, other: "measurement")-> Any:
        equal = False
        if self.iValue == other.iValue:
            if self.dValue <= other.dValue:
                equal = True
        return equal
    def __gt__(self, other: "measurement")-> Any:
        equal = False
        if self.iValue == other.iValue:
            if self.dValue > other.dValue:
                equal = True
        return equal
    def __ge__(self, other: "measurement")-> Any:
        equal = False
        if self.iValue == other.iValue:
            if self.dValue >= other.dValue:
                equal = True
        return equal
    #negation, add, right addition, sub, right sub
    def __neg__(self) -> "measurement":
        return measurement(
            self.iValue,
            -self.dValue
        )
    def __add__(self, other: Any) -> "measurement":
        if isinstance(other, measurement):
            if not self.iValue == other.iValue:
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurement(
                    self.iValue,
                    self.dValue + other.dValue
                )
        else:
            return measurement(
                self.iValue,
                self.dValue + other
            )
    def __radd__(self, other: Any) -> "measurement":
        if isinstance(other, measurement):
            if not self.iValue == other.iValue:
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurement(
                    self.iValue,
                    other.dValue + self.dValue
                )
        else:
            return measurement(
                self.iValue,
                other + self.dValue
            )

    def __sub__(self, other: Any) -> "measurement":
        if isinstance(other, measurement):
            if not self.iValue == other.iValue:
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurement(
                    self.iValue,
                    self.dValue - other.dValue
                )
        else:
            return measurement(
                self.iValue,
                self.dValue - other
            )
    def __rsub__(self, other: Any) -> "measurement":
        if isinstance(other, measurement):
            if not self.iValue == other.iValue:
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurement(
                    self.iValue,
                    other.dValue - self.dValue
                )
        else:
            return measurement(
                self.iValue,
                other - self.dValue
            )
    #mult, right mult, truediv, right true div
    def __mul__(self, other: Any) -> "measurement":
        if isinstance(other, measurement):
            if not self.iValue == other.iValue:
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurement(
                    self.iValue,
                    self.dValue * other.dValue
                )
        else:
            return measurement(
                self.iValue,
                self.dValue * other
            )            
    def __rmul__(self, other: Any) -> "measurement":
        if isinstance(other, measurement):
            if not self.iValue == other.iValue:
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurement(
                    self.iValue,
                    other.dValue * self.dValue
                )
        else:
            return measurement(
                self.iValue,
                other * self.dValue
            )
    def __truediv__( self, other: Any) -> "measurement":
        if isinstance(other, measurement):
            if not self.iValue == other.iValue:
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurement(
                    self.iValue,
                    self.dValue / other.dValue
                )
        else:
            return measurement(
                self.iValue,
                self.dValue / other
            )
    def __rtruediv__(self, other: Any) -> "measurement":
        if isinstance(other, measurement):
            if not self.iValue == other.iValue:
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurement(
                    self.iValue,
                    other.dValue / self.dValue
                )
        else:
            return measurement(
                self.iValue,
                other / self.dValue
            )
    #pow, root, sqrt
    def __pow__(self, other: float) -> "measurement":
        return measurement(
            self.iValue,
            self.dValue**other
        )
    def root(self, other:float) -> "measurement":
        return measurement(
            self.iValue,
            self.dValue**(1.0/other)
        )        
    def sqrt(self) -> "measurement":
        return self.root(2)
