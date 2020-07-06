from ilaff.units.quantity import Quantity

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, eq=False, order=False)
class measurementJack:
    iValue: Quantity
    dValue: Quantity
    jackDV: Quantity


    def jackerr(self) -> Quantity:
        """


 subroutine JackKnife_wp ( c, err, bias )
    real(kind=WP), dimension(0:), intent(in)  :: c
    real(kind=WP),                intent(out) :: err
    real(kind=WP), optional,      intent(out) :: bias
    real(kind=WP)                             :: avg
    avg = sum(c(1:))/real(size(c)-1, kind=WP)
    err = sqrt(sum( ( c(1:) - avg )**2 )*real(size(c)-2, kind=WP)/real(size(c)-1, kind=WP))
    if (present(bias)) bias = c(0) - avg
  end subroutine JackKnife_wp
        
        """


        """
        #This form works if jackDV[:] means all jackknifes for taht time slice
        jerr = np.empty( [ len(self.iValue) ] )
        for tt in range(0,len(self.iValue) ):
            #non ufunc versions
            avg = np.mean( self.jackDV[tt].value )
            err = np.sqrt( ( np.sum( ( self.jackDV[tt].value - avg )**2.0 ) ) * ( float(len(self.jackDV[tt])-1)/float(len(self.jackDV[tt])) ) )
            jerr[tt] = err
        return Quantity( jerr, self.jackDV[1].dimension, self.jackDV[1].scale )
        """

        tempJack = np.flipud( np.rot90(self.jackDV.value) )

        jerr = np.empty( [ len(self.iValue) ] )
        for tt in range(0,len(self.iValue) ):
            #non ufunc versions
            avg = np.mean( tempJack[tt] )
            err = np.sqrt( ( np.sum( ( tempJack[tt] - avg )**2.0 ) ) * ( float(len(tempJack[tt])-1)/float(len(tempJack[tt])) ) )
            jerr[tt] = err
        return Quantity( jerr, self.jackDV[1].dimension, self.jackDV[1].scale )        




    #=,>,<,>=,<= comparators
    def __eq__(self, other: "measurementJack") -> Any:
        equal = False
        if (self.iValue == other.iValue).all():
            if (self.dValue == other.dValue).all():
                print('TODO: TEST JACKKNIFES')
                equal = True
        return equal
    def __lt__(self, other: "measurementJack") -> Any:
        equal = False
        if (self.iValue == other.iValue).all():
            if (self.dValue < other.dValue).all():
                print('TODO: TEST JACKKNIFES')
                equal = True
        return equal
    def __le__(self, other: "measurementJack") -> Any:
        equal = False
        if (self.iValue == other.iValue).all():
            if (self.dValue <= other.dValue).all():
                print('TODO: TEST JACKKNIFES')
                equal = True
        return equal
    def __gt__(self, other: "measurementJack") -> Any:
        equal = False
        if (self.iValue == other.iValue).all():
            if (self.dValue > other.dValue).all():
                print('TODO: TEST JACKKNIFES')
                equal = True
        return equal
    def __ge__(self, other: "measurementJack") -> Any:
        equal = False
        if (self.iValue == other.iValue).all():
            if (self.dValue >= other.dValue).all():
                print('TODO: TEST JACKKNIFES')
                equal = True
        return equal
    #negation, add, right addition, sub, right sub
    def __neg__(self) -> "measurementJack":
        return measurementJack(
            self.iValue,
            -self.dValue,
            -self.jackDV
        )
    def __add__(self, other: "measurementJack") -> "measurementJack":
        if isinstance(other, measurementJack):
            if not (self.iValue == other.iValue).all():
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurementJack(
                    self.iValue,
                    self.dValue + other.dValue,
                    self.jackDV + other.jackDV
                )
        else:
            return measurementJack(
                self.iValue,
                self.dValue + other,
                self.jackDV + other
            )
    def __radd__(self, other: Any) -> "measurementJack":
        if isinstance(other, measurementJack):
            if not (self.iValue == other.iValue).all():
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurementJack(
                    self.iValue,
                    other.dValue + self.dValue,
                    other.jackDV + self.jackDV
                )
        else:
            return measurementJack(
                self.iValue,
                other + self.dValue,
                other + self.jackDV
            )

    def __sub__(self, other: Any) -> "measurementJack":
        if isinstance(other, measurementJack):
            if not (self.iValue == other.iValue).all():
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurementJack(
                    self.iValue,
                    self.dValue - other.dValue,
                    self.jackDV - other.jackDV
                )
        else:
            return measurementJack(
                self.iValue,
                self.dValue - other,
                self.jackDV - other
            )
    def __rsub__(self, other: Any) -> "measurementJack":
        if isinstance(other, measurementJack):
            if not (self.iValue == other.iValue).all():
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurementJack(
                    self.iValue,
                    other.dValue - self.dValue ,
                    other.jackDV - self.jackDV
                )
        else:
            return measurementJack(
                self.iValue,
                other - self.dValue,
                other - self.jackDV
            )
    #mult, right mult, truediv, right true div
    def __mul__(self, other: Any) -> "measurementJack":
        if isinstance(other, measurementJack):
            if not (self.iValue == other.iValue).all():
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurementJack(
                    self.iValue,
                    self.dValue * other.dValue,
                    self.jackDV * other.jackDV
                )
        else:
            return measurementJack(
                self.iValue,
                self.dValue * other,
                self.jackDV * other
            )            
    def __rmul__(self, other: Any) -> "measurementJack":
        if isinstance(other, measurementJack):
            if not (self.iValue == other.iValue).all():
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurementJack(
                    self.iValue,
                    other.dValue * self.dValue,
                    other.jackDV * self.jackDV
                )
        else:
            return measurementJack(
                self.iValue,
                other * self.dValue,
                other * self.jackDV
            )
    def __truediv__( self, other: Any) -> "measurementJack":
        if isinstance(other, measurementJack):
            if not (self.iValue == other.iValue).all():
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurementJack(
                    self.iValue,
                    self.dValue / other.dValue,
                    self.jackDV / other.jackDV
                )
        else:
            return measurementJack(
                self.iValue,
                self.dValue / other,
                self.jackDV / other
            )
    def __rtruediv__(self, other: Any) -> "measurementJack":
        if isinstance(other, measurementJack):
            if not (self.iValue == other.iValue).all():
                print('TODO: Format the printing of two iValues')
                raise ValueError(
                    "Can't add values: Incompatible independent variables"
                )
            else:
                return measurementJack(
                    self.iValue,
                    other.dValue / self.dValue,
                    other.jackDV / self.jackDV
                )
        else:
            return measurementJack(
                self.iValue,
                other / self.dValue,
                other / self.jackDV
            )
    #pow, root, sqrt
    def __pow__(self, other: float) -> "measurementJack":
        return measurementJack(
            self.iValue,
            self.dValue**other,
            self.jackDV**other
        )
    def root(self, other:float) -> "measurementJack":
        return measurementJack(
            self.iValue,
            self.dValue**(1.0/other),
            self.jackDV**(1.0/other)
        )        
    def sqrt(self) -> "measurementJack":
        return self.root(2)
