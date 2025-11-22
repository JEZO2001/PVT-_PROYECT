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

#%%
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
presiones1 = np.arange(0, 5000, 100)  # De 0 a 5000 psi
Pb_ejemplo1 = 2500.0  # Presión de burbuja (psi)
Rs_b_ejemplo1 = 500.0  # Rs en la burbuja (scf/stb)
gg1 = 0.7  # Gravedad gas
go1 = 0.85  # Gravedad petróleo (aprox 35 API)
Temp1 = 180.0  # Temperatura (°F)
Co_ejemplo1 = 1.5e-5  # Compresibilidad (1/psi)

# --- CÁLCULO ---
bo_resultados = calcular_bo_standing(presiones1, Pb_ejemplo1, Rs_b_ejemplo1, gg1, go1, Temp1,
                                     Co_ejemplo1)

# --- GRAFICAR ---
plt.figure(figsize=(10, 6))
plt.plot(presiones1, bo_resultados, label='Bo Calculado', color='blue', linewidth=2)

# Añadir línea vertical para la Pb
plt.axvline(x=Pb_ejemplo1, color='red', linestyle='--',
            label=f'Punto de Burbuja ({Pb_ejemplo1} psi)')

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



#%%
import numpy as np
import matplotlib.pyplot as plt

def calcular_rs_standing_perfil(P_array, Pb, Rs_b, gamma_g, gamma_o, T):
    """
    Calcula el perfil de Rs (Relación Gas-Petróleo de solución) para un rango de presiones.

    Parámetros:
    -----------
    P_array : array-like, presiones a evaluar (psi).
    Pb      : float, Presión de burbuja (psi).
    Rs_b    : float, Rs en el punto de burbuja (scf/stb). Es el valor máximo.
    gamma_g : float, Gravedad específica del gas.
    gamma_o : float, Gravedad específica del petróleo.
    T       : float, Temperatura (°F).

    Retorna:
    --------
    Rs_array : np.array, Valores calculados de Rs.
    """
    Rs_list = []

    # 1. Calcular gravedad API
    API = (141.5 / gamma_o) - 131.5

    # 2. Calcular el exponente 'x', que es constante para una T y API dadas
    # x = 0.0125 * API - 0.00091 * T
    x_exponent = 0.0125 * API - 0.00091 * T

    # 3. Iterar sobre las presiones
    for P in P_array:
        if P < Pb:
            # --- Región Saturada (P < Pb) ---
            # El Rs aumenta con la presión según la correlación.

            # Fórmula: Rs = gamma_g * [ (P/18.2 + 1.4) * 10^x ]^1.2048
            term_pressure = (P / 18.2) + 1.4
            term_combined = term_pressure * (10**x_exponent)

            Rs_calc = gamma_g * (term_combined ** 1.2048)

            # Importante: Por inconsistencias en los datos de entrada vs correlación,
            # el Rs calculado no debe superar el Rs_b conocido.
            Rs = min(Rs_calc, Rs_b)

        else:
            # --- Región Sub-saturada (P >= Pb) ---
            # El Rs es constante e igual al Rs en el punto de burbuja.
            Rs = Rs_b

        Rs_list.append(Rs)

    return np.array(Rs_list)

# --- DATOS DE EJEMPLO (Coherentes con el ejemplo anterior) ---
presiones_rs = np.arange(0, 5000, 50) # Presiones de 0 a 5000 psi
Pb_ej = 2500.0      # Presión de burbuja (psi)
Rs_b_ej = 500.0     # Rs máximo en la burbuja (scf/stb)
gg_ej = 0.7         # Gravedad gas
go_ej = 0.85        # Gravedad petróleo
T_ej = 180.0        # Temperatura (°F)

# --- CÁLCULO ---
rs_resultados = calcular_rs_standing_perfil(presiones_rs, Pb_ej, Rs_b_ej, gg_ej, go_ej, T_ej)

# --- GRAFICAR ---
plt.figure(figsize=(10, 6))

# Graficar la curva principal
plt.plot(presiones_rs, rs_resultados, label='Rs Calculado (Standing)', color='green', linewidth=2)

# Añadir marcadores visuales
plt.axvline(x=Pb_ej, color='red', linestyle='--', label=f'Presión de Burbuja ({Pb_ej} psi)')
plt.axhline(y=Rs_b_ej, color='gray', linestyle=':', label=f'Rs Máximo ({Rs_b_ej} scf/stb)')

# Anotaciones de las regiones
plt.text(Pb_ej/2, Rs_b_ej/2, 'Región Saturada\n(Gas liberándose)', ha='center', color='green')
plt.text(Pb_ej + 1000, Rs_b_ej + 20, 'Región Sub-saturada\n(Rs constante)', ha='center', color='black')

# Estilo de la gráfica
plt.title('Solubilidad del Gas ($R_s$) vs Presión', fontsize=14)
plt.xlabel('Presión (psi)', fontsize=12)
plt.ylabel('$R_s$ (scf/stb)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.ylim(bottom=0) # Asegurar que el eje Y empiece en 0

# Mostrar gráfica
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt


def calcular_densidad_oil(P_array, Pb, Rs_b, gamma_g, gamma_o, T, Co):
    """
    Calcula el perfil de Densidad del Petróleo (rho_o) vs Presión.
    """
    rho_list = []

    # 1. Calcular API y exponente para el cálculo de Rs (necesario para P < Pb)
    API = (141.5 / gamma_o) - 131.5
    x_exponent = 0.0125 * API - 0.00091 * T

    for P in P_array:
        # --- PASO PREVIO: DETERMINAR Rs ACTUAL ---
        if P < Pb:
            # Calcular Rs variable usando correlación de Standing (como vimos antes)
            term_pressure = (P / 18.2) + 1.4
            term_combined = term_pressure * (10 ** x_exponent)
            Rs_calc = gamma_g * (term_combined ** 1.2048)
            Rs_actual = min(Rs_calc, Rs_b)
        else:
            # Por encima de la burbuja, el Rs es constante
            Rs_actual = Rs_b

        # --- CÁLCULO DE DENSIDAD ---

        # Calculamos la densidad base usando la fórmula de Standing (Imagen 2)
        # Esta fórmula se usa directamente si P < Pb.
        # Si P >= Pb, esta fórmula nos da la densidad EN el punto de burbuja (rho_ob).

        # Numerador: (62.4 * go + 0.0136 * Rs * gg)
        num = (62.4 * gamma_o) + (0.0136 * Rs_actual * gamma_g)

        # Denominador: 0.972 + 0.000147 * [Rs * (gg/go)^0.25 + 1.25*T]^1.175
        # Nota: Tu imagen usa exponente 0.25 para la relación de gravedades.
        term_den = Rs_actual * ((gamma_g / gamma_o) ** 0.25) + (1.25 * T)
        den = 0.972 + 0.000147 * (term_den ** 1.175)

        rho_standing = num / den

        if P < Pb:
            # REGIÓN SATURADA
            rho = rho_standing
        else:
            # REGIÓN SUB-SATURADA (Imagen 1)
            # rho = rho_ob * exp(Co * (P - Pb))
            # Aquí 'rho_standing' calculado con Rs_b es nuestro 'rho_ob'
            rho_ob = rho_standing
            rho = rho_ob * np.exp(Co * (P - Pb))

        rho_list.append(rho)

    return np.array(rho_list)


# --- DATOS DE ENTRADA ---
presiones = np.arange(100, 5000,
                      100)  # Evitamos 0 absoluto para no romper logaritmos si los hubiera
Pb_ej = 2500.0  # psi
Rs_b_ej = 500.0  # scf/stb
gg_ej = 0.7  # Gravedad gas
go_ej = 0.85  # Gravedad aceite
T_ej = 180.0  # °F
Co_ej = 1.5e-5  # 1/psi

# --- EJECUCIÓN ---
rho_resultados = calcular_densidad_oil(presiones, Pb_ej, Rs_b_ej, gg_ej, go_ej, T_ej,
                                       Co_ej)

# --- GRÁFICA ---
plt.figure(figsize=(10, 6))
plt.plot(presiones, rho_resultados, label='Densidad Petróleo (Standing)',
         color='purple', linewidth=2)

# Línea vertical en Pb
plt.axvline(x=Pb_ej, color='red', linestyle='--',
            label=f'Punto de Burbuja ({Pb_ej} psi)')

# Anotaciones para entender la física
plt.text(Pb_ej - 1500, min(rho_resultados) * 1.01,
         'Gas liberándose\n(Líquido remanente más pesado)', fontsize=9, color='purple')
plt.text(Pb_ej + 500, max(rho_resultados) * 0.99, 'Compresión del Líquido', fontsize=9,
         color='purple')

plt.title('Densidad del Petróleo vs Presión', fontsize=14)
plt.xlabel('Presión (psi)', fontsize=12)
plt.ylabel('Densidad ($lb/ft^3$)', fontsize=12)
plt.grid(True, alpha=0.6)
plt.legend()

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt


def calcular_viscosidad_oil(P_array, Pb, Rs_b, gamma_g, gamma_o, T):
    """
    Calcula el perfil de Viscosidad del Petróleo (mu_o) vs Presión.
    """
    mu_list = []

    # --- 1. CALCULAR VISCOSIDAD DEL PETRÓLEO MUERTO (mu_od) ---
    # Usamos la correlación de Beggs & Robinson (1975) ya que es el input
    # necesario para las fórmulas de tus imágenes.

    API = (141.5 / gamma_o) - 131.5

    # Z = 3.0324 - 0.02023 * API
    # Y = 10^Z
    # X = Y * T^(-1.163)  (Nota: T en Farenheit)
    # mu_od = 10^X - 1

    Z = 3.0324 - 0.02023 * API
    Y = 10 ** Z
    X = Y * (T ** -1.163)
    mu_od = (10 ** X) - 1

    # Exponente x para el cálculo auxiliar de Rs (Standing)
    x_exponent_rs = 0.0125 * API - 0.00091 * T

    for P in P_array:
        # --- 2. DETERMINAR Rs ACTUAL ---
        if P < Pb:
            # Rs varía con la presión (Standing)
            term_pressure = (P / 18.2) + 1.4
            term_combined = term_pressure * (10 ** x_exponent_rs)
            Rs_calc = gamma_g * (term_combined ** 1.2048)
            Rs_actual = min(Rs_calc, Rs_b)
        else:
            Rs_actual = Rs_b

        # --- 3. CALCULAR VISCOSIDAD SATURADA (Imagen 1) ---
        # Esta fórmula se usa para P < Pb con Rs actual,
        # o para encontrar mu_ob (viscosidad en burbuja) usando Rs_b.

        # a = 10.715 * (Rs + 100)^-0.515
        a = 10.715 * ((Rs_actual + 100) ** -0.515)

        # b = 5.44 * (Rs + 150)^-0.338
        b = 5.44 * ((Rs_actual + 150) ** -0.338)

        # mu_sat = a * (mu_od)^b
        mu_sat = a * (mu_od ** b)

        if P < Pb:
            # REGIÓN SATURADA
            # La viscosidad depende del gas disuelto actual
            mu = mu_sat
        else:
            # REGIÓN SUB-SATURADA (Imagen 2)
            # Primero definimos la viscosidad en la burbuja (mu_ob)
            # Recalculamos mu_sat pero con Rs_b fijo (ya que arriba usamos Rs_actual que es igual a Rs_b aqui)
            mu_ob = mu_sat

            # Calculamos el exponente m (Imagen 2)
            # m = 2.6 * P^1.187 * exp(-11.513 - 8.98e-5 * P)
            exponent_term = -11.513 - (8.98e-5 * P)
            m = 2.6 * (P ** 1.187) * np.exp(exponent_term)

            # Fórmula final: mu = mu_ob * (P/Pb)^m
            mu = mu_ob * ((P / Pb) ** m)

        mu_list.append(mu)

    return np.array(mu_list)


# --- DATOS DE ENTRADA ---
presiones = np.arange(100, 5000, 50)
Pb_ej = 2500.0  # psi
Rs_b_ej = 500.0  # scf/stb
gg_ej = 0.7  # Gravedad gas
go_ej = 0.85  # Gravedad aceite (aprox 35 API)
T_ej = 180.0  # °F

# --- CÁLCULO ---
mu_resultados = calcular_viscosidad_oil(presiones, Pb_ej, Rs_b_ej, gg_ej, go_ej, T_ej)

# --- GRÁFICA ---
plt.figure(figsize=(10, 6))
plt.plot(presiones, mu_resultados, label=r'Viscosidad (${\mu_o}$)', color='brown',
         linewidth=2)

# Línea vertical en Pb
plt.axvline(x=Pb_ej, color='red', linestyle='--',
            label=f'Punto de Burbuja ({Pb_ej} psi)')

# Anotaciones didácticas
mu_min = min(mu_resultados)
mu_max = max(mu_resultados)

plt.text(Pb_ej - 1500, (mu_max + mu_min) / 2,
         'Saturado:\nGas liberándose = Mayor Viscosidad',
         fontsize=9, color='brown', ha='center')
plt.text(Pb_ej + 1000, mu_min + (mu_max - mu_min) * 0.1,
         'Sub-Saturado:\nCompresión = Ligero aumento',
         fontsize=9, color='black', ha='center')

plt.title('Viscosidad del Petróleo vs Presión', fontsize=14)
plt.xlabel('Presión (psi)', fontsize=12)
plt.ylabel('Viscosidad (cp)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()

plt.show()