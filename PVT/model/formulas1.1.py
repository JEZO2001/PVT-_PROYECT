import math
import numpy as np
import matplotlib.pyplot as plt

def calcular_bo_standing(P_array, Pb, Rs_b, gamma_g, gamma_o, T, Co):
    """
    Calcula y grafica el perfil de Bo (Factor Volumétrico del Petróleo).
    """
    Bo_list = []

    for P in P_array:
        if P < Pb:
            # --- REGIÓN SATURADA (P < Pb) ---
            API = (141.5 / gamma_o) - 131.5
            exponent = 0.0125 * API - 0.00091 * T
            term = (P / 18.2) + 1.4
            Rs_actual = gamma_g * (term * (10 ** exponent)) ** 1.2048
            Rs_calc = min(Rs_actual, Rs_b)

            term_standing = Rs_calc * ((gamma_g / gamma_o) ** 0.5) + 1.25 * T
            Bo = 0.9759 + 0.000120 * (term_standing ** 1.2)
        else:
            # --- REGIÓN SUB-SATURADA (P >= Pb) ---
            term_standing_b = Rs_b * ((gamma_g / gamma_o) ** 0.5) + 1.25 * T
            Bob = 0.9759 + 0.000120 * (term_standing_b ** 1.2)
            Bo = Bob * np.exp(Co * (Pb - P))

        Bo_list.append(Bo)

    Bo_array = np.array(Bo_list)

    # --- GRAFICAR ---
    plt.figure(figsize=(10, 6))
    plt.plot(P_array, Bo_array, label='Bo Calculado', color='blue', linewidth=2)
    plt.axvline(x=Pb, color='red', linestyle='--', label=f'Punto de Burbuja ({Pb} psi)')

    plt.title('Factor Volumétrico de Formación (Bo) vs Presión', fontsize=14)
    plt.xlabel('Presión (psi)', fontsize=12)
    plt.ylabel('Bo (rb/stb)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    return Bo_array


def calcular_rs_standing_perfil(P_array, Pb, Rs_b, gamma_g, gamma_o, T):
    """
    Calcula y grafica el perfil de Rs (Relación Gas-Petróleo de solución).
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

    Rs_array = np.array(Rs_list)

    # --- GRAFICAR ---
    plt.figure(figsize=(10, 6))
    plt.plot(P_array, Rs_array, label='Rs Calculado', color='green', linewidth=2)
    plt.axvline(x=Pb, color='red', linestyle='--', label=f'Punto de Burbuja ({Pb} psi)')
    plt.axhline(y=Rs_b, color='gray', linestyle=':',
                label=f'Rs Máximo ({Rs_b} scf/stb)')

    plt.text(Pb / 2, Rs_b / 2, 'Región Saturada', ha='center', color='green')
    plt.text(Pb + (np.max(P_array) - Pb) / 2, Rs_b + (Rs_b * 0.05),
             'Región Sub-saturada', ha='center', color='black')

    plt.title('Solubilidad del Gas (Rs) vs Presión', fontsize=14)
    plt.xlabel('Presión (psi)', fontsize=12)
    plt.ylabel('Rs (scf/stb)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.ylim(bottom=0)
    plt.show()

    return Rs_array


def calcular_densidad_oil(P_array, Pb, Rs_b, gamma_g, gamma_o, T, Co):
    """
    Calcula y grafica el perfil de Densidad del Petróleo.
    """
    rho_list = []
    API = (141.5 / gamma_o) - 131.5
    x_exponent = 0.0125 * API - 0.00091 * T

    for P in P_array:
        if P < Pb:
            term_pressure = (P / 18.2) + 1.4
            term_combined = term_pressure * (10 ** x_exponent)
            Rs_calc = gamma_g * (term_combined ** 1.2048)
            Rs_actual = min(Rs_calc, Rs_b)
        else:
            Rs_actual = Rs_b

        num = (62.4 * gamma_o) + (0.0136 * Rs_actual * gamma_g)
        term_den = Rs_actual * ((gamma_g / gamma_o) ** 0.25) + (1.25 * T)
        den = 0.972 + 0.000147 * (term_den ** 1.175)
        rho_standing = num / den

        if P < Pb:
            rho = rho_standing
        else:
            rho_ob = rho_standing
            rho = rho_ob * np.exp(Co * (P - Pb))

        rho_list.append(rho)

    rho_array = np.array(rho_list)

    # --- GRAFICAR ---
    plt.figure(figsize=(10, 6))
    plt.plot(P_array, rho_array, label='Densidad Petróleo', color='purple', linewidth=2)
    plt.axvline(x=Pb, color='red', linestyle='--', label=f'Punto de Burbuja ({Pb} psi)')

    plt.title('Densidad del Petróleo vs Presión', fontsize=14)
    plt.xlabel('Presión (psi)', fontsize=12)
    plt.ylabel('Densidad (lb/ft^3)', fontsize=12)
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.show()

    return rho_array


def calcular_viscosidad_oil(P_array, Pb, Rs_b, gamma_g, gamma_o, T):
    """
    Calcula y grafica el perfil de Viscosidad del Petróleo.
    """
    mu_list = []
    API = (141.5 / gamma_o) - 131.5
    Z = 3.0324 - 0.02023 * API
    Y = 10 ** Z
    X = Y * (T ** -1.163)
    mu_od = (10 ** X) - 1
    x_exponent_rs = 0.0125 * API - 0.00091 * T

    for P in P_array:
        if P < Pb:
            term_pressure = (P / 18.2) + 1.4
            term_combined = term_pressure * (10 ** x_exponent_rs)
            Rs_calc = gamma_g * (term_combined ** 1.2048)
            Rs_actual = min(Rs_calc, Rs_b)
        else:
            Rs_actual = Rs_b

        a = 10.715 * ((Rs_actual + 100) ** -0.515)
        b = 5.44 * ((Rs_actual + 150) ** -0.338)
        mu_sat = a * (mu_od ** b)

        if P < Pb:
            mu = mu_sat
        else:
            mu_ob = mu_sat
            exponent_term = -11.513 - (8.98e-5 * P)
            m = 2.6 * (P ** 1.187) * np.exp(exponent_term)
            mu = mu_ob * ((P / Pb) ** m)

        mu_list.append(mu)

    mu_array = np.array(mu_list)

    # --- GRAFICAR ---
    plt.figure(figsize=(10, 6))
    plt.plot(P_array, mu_array, label='Viscosidad', color='brown', linewidth=2)
    plt.axvline(x=Pb, color='red', linestyle='--', label=f'Punto de Burbuja ({Pb} psi)')

    plt.title('Viscosidad del Petróleo vs Presión', fontsize=14)
    plt.xlabel('Presión (psi)', fontsize=12)
    plt.ylabel('Viscosidad (cp)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    return mu_array