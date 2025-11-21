import math
import numpy as np
import matplotlib.pyplot as plt


#%% def pb(Rs, yg, T, API):
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

def calcular_bo_standing(P_array, Pb, Rs_b, gamma_g, gamma_o, T, Co):
    """
    Calcula el perfil de Bo (Factor Volumétrico del Petróleo) para un rango de presiones
    utilizando la correlación de Standing y el ajuste por compresibilidad.

    Parámetros:
    -----------
    P_array : array-like
        Array de presiones a evaluar (psi).
    Pb : float
        Presión de burbuja (psi).
    Rs_b : float
        Relación Gas-Petróleo de solución en el punto de burbuja (scf/stb).
    gamma_g : float
        Gravedad específica del gas (aire = 1).
    gamma_o : float
        Gravedad específica del petróleo (agua = 1).
    T : float
        Temperatura en grados Fahrenheit (°F).
    Co : float
        Compresibilidad del petróleo (psi^-1).

    Retorna:
    --------
    Bo_array : np.array
        Valores calculados de Bo.
    """

    Bo_array = []

    # Constantes para simplificar la lectura de la fórmula de Standing
    # B_o = 0.9759 + 0.000120 * [Rs * (gg/go)^0.5 + 1.25*T]^1.2

    for P in P_array:
        if P < Pb:
            # --- REGIÓN SATURADA (P < Pb) ---
            # Nota: Por debajo de Pb, el gas sale de solución, por lo que Rs disminuye.
            # Aproximación de Standing para Rs en función de la Presión actual (P):
            # Rs = gamma_g * ((P / 18.2 + 1.4) * 10^(0.0125*API - 0.00091*T))^1.2048
            # Para simplificar y usar tus inputs, escalaremos el Rs linealmente o
            # usamos la correlación inversa de Standing. Aquí usaré la aproximación estándar:

            # Calculamos API para la fórmula auxiliar de Rs
            API = (141.5 / gamma_o) - 131.5

            # Estimar Rs actual a la presión P (Correlación inversa de Standing)
            exponent = 0.0125 * API - 0.00091 * T
            term = (P / 18.2) + 1.4
            Rs_actual = gamma_g * (term * (10 ** exponent)) ** 1.2048

            # Nos aseguramos que no supere el Rs_b de entrada (por errores de redondeo)
            Rs_calc = min(Rs_actual, Rs_b)

            # Aplicamos FÓRMULA DE LA IMAGEN 2
            term_standing = Rs_calc * ((gamma_g / gamma_o) ** 0.5) + 1.25 * T
            Bo = 0.9759 + 0.000120 * (term_standing ** 1.2)

        else:
            # --- REGIÓN SUB-SATURADA (P >= Pb) ---
            # 1. Calcular Bob (Bo en el punto de burbuja)
            term_standing_b = Rs_b * ((gamma_g / gamma_o) ** 0.5) + 1.25 * T
            Bob = 0.9759 + 0.000120 * (term_standing_b ** 1.2)

            # 2. Aplicar FÓRMULA DE LA IMAGEN 1 (Ajuste por compresibilidad)
            # Bo = Bob * exp(Co * (Pb - P))
            # Nota: Si P > Pb, (Pb - P) es negativo, por lo que Bo disminuye.
            Bo = Bob * np.exp(Co * (Pb - P))

        Bo_array.append(Bo)

    return np.array(Bo_array)


# --- DATOS DE EJEMPLO ----
presiones = np.arange(0, 5000, 100)  # De 0 a 5000 psi
Pb_ejemplo = 2500.0  # Presión de burbuja (psi)
Rs_b_ejemplo = 500.0  # Rs en la burbuja (scf/stb)
gg = 0.7  # Gravedad gas
go = 0.85  # Gravedad petróleo (aprox 35 API)
Temp = 180.0  # Temperatura (°F)
Co_ejemplo = 1.5e-5  # Compresibilidad (1/psi)

# --- CÁLCULO ---
bo_resultados = calcular_bo_standing(presiones, Pb_ejemplo, Rs_b_ejemplo, gg, go, Temp,
                                     Co_ejemplo)

# --- GRAFICAR ---
plt.figure(figsize=(10, 6))
plt.plot(presiones, bo_resultados, label='Bo Calculado', color='blue', linewidth=2)

# Añadir línea vertical para la Pb
plt.axvline(x=Pb_ejemplo, color='red', linestyle='--',
            label=f'Punto de Burbuja ({Pb_ejemplo} psi)')

# Etiquetas y Estilo
plt.title(
    'Factor Volumétrico de Formación del Petróleo (Bo) vs Presión\n(Correlación de Standing & Compresibilidad)',
    fontsize=14)
plt.xlabel('Presión (psi)', fontsize=12)
plt.ylabel('Bo (rb/stb)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

# Mostrar gráfica
plt.show()

import numpy as np
import matplotlib.pyplot as plt


def calcular_Rs_standing(P, T, API, gamma_g):
    """
    Calcula el Rs usando la correlación de Standing.

    Parámetros:
    P       : Presión (psia)
    T       : Temperatura (Fahrenheit)
    API     : Gravedad API del petróleo
    gamma_g : Gravedad específica del gas
    """
    # Cálculo del exponente x según la segunda fórmula de la imagen
    x = 0.0125 * API - 0.00091 * T

    # Cálculo de Rs según la primera fórmula
    # Rs = gamma_g * [ (P/18.2 + 1.4) * 10^x ] ^ 1.2048
    term_inner = (P / 18.2) + 1.4
    term_bracket = term_inner * (10 ** x)
    Rs = gamma_g * (term_bracket ** 1.2048)

    return Rs


# --- 1. Definir los datos de entrada ---
# Rango de presiones de 0 a 5000 psia (puedes ajustar esto)
presiones = np.linspace(14.7, 5000, 100)

# Propiedades fijas del fluido (Ejemplo típico)
temp_F = 200  # Temperatura en °F
gravedad_api = 35  # °API
gravedad_gas = 0.7  # Gravedad específica del gas

# --- 2. Calcular Rs ---
rs_valores = calcular_Rs_standing(presiones, temp_F, gravedad_api, gravedad_gas)

# --- 3. Graficar con Matplotlib ---
plt.figure(figsize=(10, 6))  # Tamaño de la figura

plt.plot(presiones, rs_valores,
         label=f'API={gravedad_api}, T={temp_F}°F, $\gamma_g$={gravedad_gas}',
         color='blue', linewidth=2)

# Formato del gráfico
plt.title('Correlación de Standing: Solubilidad del Gas ($R_s$) vs Presión',
          fontsize=14)
plt.xlabel('Presión (psia)', fontsize=12)
plt.ylabel('Rs (pc/bbl)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.minorticks_on()

# Mostrar gráfico
plt.show()