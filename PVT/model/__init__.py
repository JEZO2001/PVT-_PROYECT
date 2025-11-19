import math


def pb(Rs, yg, T, API):
    """
    Rs: Solubilidad del gas en petróleo (ft^3/bbl).
    yg: Gravedad específica del gas.
    T: Temperatura (°F).
    API: Gravedad API del petróleo.
    """

    # Calcular A
    A = ((Rs / yg) ** 0.816) * ((T ** 0.130) / (API ** 0.989))
    log_A = math.log10(A)

    # Calcular Pb
    Pb = 1.7669 + 1.7447 * log_A - 0.30218 * (log_A) ** 2

    return Pb