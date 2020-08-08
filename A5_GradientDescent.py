import decimal as dc

## Learning rate
n = dc.Decimal(0.1)

## Initialising wieghts, errors and runs
u = dc.Decimal(1)
v = dc.Decimal(1)
runs = 0
E = (u * dc.Decimal.exp(v) - dc.Decimal(2) * v * dc.Decimal.exp(-u)) ** dc.Decimal(2)

## Iterating to meet the given conditiion
while (E > (10 ** -14)):
    uDer = dc.Decimal(2) * (dc.Decimal.exp(v) + dc.Decimal(2) * v * dc.Decimal.exp(-u)) * (u * dc.Decimal.exp(v) - dc.Decimal(2) * v * dc.Decimal.exp(-u))
    vDer = dc.Decimal(2) * (dc.Decimal(u) * dc.Decimal.exp(v) - dc.Decimal(2) * dc.Decimal.exp(-u)) * (u * dc.Decimal.exp(v) - dc.Decimal(2) * v * dc.Decimal.exp(-u))
    u = u - dc.Decimal(n * uDer)
    v = v - dc.Decimal(n * vDer)
    E = (u * dc.Decimal.exp(v) - dc.Decimal(2) * v * dc.Decimal.exp(-u)) ** dc.Decimal(2)
    runs = runs + 1
    print(runs)
    print(E*(10**14))
