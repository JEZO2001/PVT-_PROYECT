import numpy as np
import math


# --- 1. Factor de Compresibilidad (Co) ---
def calcular_co_petrosky_vasquez(P_array, Pb, Rs_b, gamma_g, gamma_o, T, API, PR):
    """
    Calcula el Factor de Compresibilidad del Petróleo (Co).
    Usa Petrosky y Farshad (P > Pb) y asume 0 para saturado (P <= Pb).

    Parámetros:
    P_array (array): Vector de Presiones, psi.
    Pb (float): Presión de Burbuja, psi.
    Rs_b (float): RGP de Burbuja, scf/stb.
    gamma_g (float): Densidad Específica del Gas (aire=1).
    gamma_o (float): Densidad Específica del Petróleo (agua=1).
    T (float): Temperatura, °F.
    API (float): Gravedad API.
    PR (float): Presión Inicial o de Yacimiento, psi (usada por Petrosky).

    Retorna:
    Co_array (array): Factor de Compresibilidad, 1/psi.
    """
    Co_list = []
    T_R = T + 460  # Temperatura en Rankine

    for P in P_array:
        if P <= Pb:
            # --- REGIÓN SATURADA (P <= Pb) ---
            # Simplificación: Co es despreciable/cero en la región saturada
            Co = 0.0
        else:
            # --- REGIÓN SUB-SATURADA (P > Pb) ---
            # Correlación de Petrosky y Farshad
            num_term = (1.11e-7 * Rs_b) + (1.33e-5 * T) + (5.09e-3 * API)
            den_term = P * (1.27e-4 * Rs_b + 1.29e-3 * T - 3.79e-4 * API)

            # Petrosky & Farshad requiere Co en la presión de yacimiento (Pr)
            # Para fines de simplificación de Co(P) usaremos P en lugar de Pr

            # Usaremos la forma de Petrosky (para P_r) y la aplicamos al P actual
            Co_pf = (1.11e-7 * Rs_b + 1.33e-5 * T + 5.09e-3 * API) / \
                    (P * (1.27e-4 * Rs_b + 1.29e-3 * T - 3.79e-4 * API))

            Co = Co_pf

        Co_list.append(Co)

    return np.array(Co_list)


# --- 2. Factor Volumétrico de Petróleo (Bo) ---
def calcular_bo_standing(P_array, Pb, Rs_b, gamma_g, gamma_o, T, Co_array):
    """
    Calcula el perfil de Bo (Standing), usando Rs calc. para saturado
    y Co para el sub-saturado.
    """
    Bo_list = []
    API = (141.5 / gamma_o) - 131.5
    exponent = 0.0125 * API - 0.00091 * T

    # Asumimos Co_array tiene el mismo tamaño que P_array

    for i, P in enumerate(P_array):
        Co = Co_array[i]

        if P < Pb:
            # --- REGIÓN SATURADA (P < Pb) ---
            term = (P / 18.2) + 1.4
            Rs_actual = gamma_g * (term * (10 ** exponent)) ** 1.2048
            Rs_calc = min(Rs_actual, Rs_b)

            term_standing = Rs_calc * ((gamma_g / gamma_o) ** 0.5) + 1.25 * T
            Bo = 0.9759 + 0.000120 * (term_standing ** 1.2)
        else:
            # --- REGIÓN SUB-SATURADA (P >= Pb) ---
            term_standing_b = Rs_b * ((gamma_g / gamma_o) ** 0.5) + 1.25 * T
            Bob = 0.9759 + 0.000120 * (term_standing_b ** 1.2)

            # Usamos Co_array que ya tiene la compresibilidad calculada
            Bo = Bob * np.exp(Co * (Pb - P))

        Bo_list.append(Bo)

    return np.array(Bo_list)


# --- 3. Solubilidad del Gas (Rs) ---
def calcular_rs_standing_perfil(P_array, Pb, Rs_b, gamma_g, gamma_o, T):
    """
    Calcula el perfil de Rs (Standing).
    """
    Rs_list = []
    API = (141.5 / gamma_o) - 131.5
    x_exponent = 0.0125 * API - 0.00091 * T

    for P in P_array:
        if P < Pb:
            # --- Región Saturada ---
            term_pressure = (P / 18.2) + 1.4
            term_combined = term_pressure * (10 ** x_exponent)
            Rs_calc = gamma_g * (term_combined ** 1.2048)
            Rs = min(Rs_calc, Rs_b)
        else:
            # --- Región Sub-saturada ---
            Rs = Rs_b

        Rs_list.append(Rs)

    return np.array(Rs_list)


# --- 4. Densidad del Petróleo (rho_o) ---
def calcular_densidad_oil(P_array, Pb, Rs_b, gamma_g, gamma_o, T, Co_array):
    """
    Calcula el perfil de Densidad del Petróleo (Standing modificado),
    usando Bo y Co para el sub-saturado.
    """
    rho_list = []
    API = (141.5 / gamma_o) - 131.5
    x_exponent = 0.0125 * API - 0.00091 * T

    # Obtenemos Rs para la densidad saturada
    Rs_array = calcular_rs_standing_perfil(P_array, Pb, Rs_b, gamma_g, gamma_o, T)

    for i, P in enumerate(P_array):
        Rs_actual = Rs_array[i]
        Co = Co_array[i]

        # Densidad del Petróleo Saturado (rho_sat)
        num = (62.4 * gamma_o) + (0.0136 * Rs_actual * gamma_g)
        term_den = Rs_actual * ((gamma_g / gamma_o) ** 0.25) + (1.25 * T)
        den = 0.972 + 0.000147 * (term_den ** 1.175)
        rho_standing = num / den

        if P < Pb:
            # --- Región Saturada ---
            rho = rho_standing
        else:
            # --- Región Sub-saturada ---
            rho_ob = rho_standing  # Densidad a la presión de burbuja (Pb)
            # rho_o = rho_ob * exp(Co * (P - Pb))
            rho = rho_ob * np.exp(Co * (P - Pb))

        rho_list.append(rho)

    return np.array(rho_list)


# --- 5. Viscosidad del Petróleo (mu_o) ---
def calcular_viscosidad_oil(P_array, Pb, Rs_b, gamma_g, gamma_o, T):
    """
    Calcula el perfil de Viscosidad del Petróleo (Beggs y Robinson para mu_od, Vasquez y Beggs para mu_sat y sub-sat).
    """
    mu_list = []
    API = (141.5 / gamma_o) - 131.5

    # 5.1 Viscosidad del Petróleo Muerto (mu_od) - Beggs y Robinson
    Z = 3.0324 - 0.02023 * API
    Y = 10 ** Z
    X = Y * (T ** -1.163)
    mu_od = (10 ** X) - 1

    # 5.2 Obtener Rs_actual para mu_sat
    Rs_array = calcular_rs_standing_perfil(P_array, Pb, Rs_b, gamma_g, gamma_o, T)

    for i, P in enumerate(P_array):
        Rs_actual = Rs_array[i]

        # Viscosidad del Petróleo Saturado (mu_sat) - Vasquez y Beggs
        a = 10.715 * ((Rs_actual + 100) ** -0.515)
        b = 5.44 * ((Rs_actual + 150) ** -0.338)
        mu_sat = a * (mu_od ** b)

        if P < Pb:
            # --- Región Saturada ---
            mu = mu_sat
        else:
            # --- Región Sub-saturada (P >= Pb) ---
            mu_ob = mu_sat
            # Coeficiente 'm' de compresibilidad de la viscosidad (Vasquez y Beggs)
            exponent_term = -11.513 - (8.98e-5 * P)
            m = 2.6 * (P ** 1.187) * np.exp(exponent_term)
            mu = mu_ob * ((P / Pb) ** m)

        mu_list.append(mu)

    return np.array(mu_list)