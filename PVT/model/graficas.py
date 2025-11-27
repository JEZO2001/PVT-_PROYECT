# main_graficar.py

import numpy as np
import matplotlib.pyplot as plt
import random
from correlaciones_pvt import (
    calcular_co_petrosky_vasquez,
    calcular_bo_standing,
    calcular_rs_standing_perfil,
    calcular_densidad_oil,
    calcular_viscosidad_oil
)

# --- 1. Datos de Yacimiento (Datos del Profesor) ---
Pb = 3970.0 # psi
Rs_b = 1124.0 # scf/stb
API = 38.982
sg_gas = 0.65
Pr = 4409.0 # psia (Presión Inicial del Yacimiento)
Tr = 140.0 # °F (Temperatura del Yacimiento)
gamma_g = sg_gas
gamma_o = 141.5 / (API + 131.5) # Densidad específica del petróleo (agua=1)

# --- 2. Generación de Presión Aleatoria/Rango ---
# Generamos 50 puntos de presión, incluyendo Pb y Pr, y un punto mínimo (100 psi)
P_min = 100.0
P_max = 4500.0 # Un poco más arriba de Pr para cubrir el rango de trabajo
num_puntos = 50

# Lista de presiones que aseguran incluir los puntos clave
P_array_base = list(np.linspace(P_max, P_min, num_puntos))
# Insertamos Pr y Pb para asegurar su inclusión
P_array_base.append(Pr)
P_array_base.append(Pb)
P_array = np.unique(np.sort(P_array_base))[::-1] # Eliminar duplicados, ordenar y revertir

# Si necesitas presiones verdaderamente aleatorias entre P_min y P_max:
# P_array = np.array([random.uniform(P_min, P_max) for _ in range(num_puntos)])
# P_array = np.sort(P_array)[::-1] # Ordenar de mayor a menor

print(f"Presión Máxima: {np.max(P_array):.0f} psi, Presión Mínima: {np.min(P_array):.0f} psi")

# --- 3. Cálculo del Factor de Compresibilidad (Co) ---
Co_array = calcular_co_petrosky_vasquez(P_array, Pb, Rs_b, gamma_g, gamma_o, Tr, API, Pr)

# --- 4. Cálculo de las Propiedades PVT ---
Rs_array = calcular_rs_standing_perfil(P_array, Pb, Rs_b, gamma_g, gamma_o, Tr)
Bo_array = calcular_bo_standing(P_array, Pb, Rs_b, gamma_g, gamma_o, Tr, Co_array)
rho_array = calcular_densidad_oil(P_array, Pb, Rs_b, gamma_g, gamma_o, Tr, Co_array)
mu_array = calcular_viscosidad_oil(P_array, Pb, Rs_b, gamma_g, gamma_o, Tr)


# --- 5. Generación de Gráficos Combinados ---
def generar_graficos_pvt(P, Bo, Rs, Co, rho, mu, Pb_val, Pr_val):
    """Genera 5 subplots para visualizar todas las propiedades."""
    plt.figure(figsize=(14, 12))

    # --- Gráfico 1: Factor Volumétrico (Bo) ---
    plt.subplot(3, 2, 1)
    plt.plot(P, Bo, label='Bo (Standing)', color='blue', linewidth=2)
    plt.axvline(x=Pb_val, color='red', linestyle='--', label=f'$P_b$ ({Pb_val:.0f} psi)')
    plt.title('Factor Volumétrico de Formación ($B_o$)', fontsize=12)
    plt.ylabel('$B_o$ (rb/stb)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # --- Gráfico 2: Solubilidad del Gas (Rs) ---
    plt.subplot(3, 2, 2)
    plt.plot(P, Rs, label='$R_s$ (Standing)', color='green', linewidth=2)
    plt.axvline(x=Pb_val, color='red', linestyle='--', label=f'$P_b$ ({Pb_val:.0f} psi)')
    plt.axhline(y=Rs_b, color='gray', linestyle=':', label=f'$R_{{sb}}$ ({Rs_b:.0f} scf/stb)')
    plt.title('Solubilidad del Gas ($R_s$)', fontsize=12)
    plt.ylabel('$R_s$ (scf/stb)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right')

    # --- Gráfico 3: Viscosidad (mu_o) ---
    plt.subplot(3, 2, 3)
    plt.plot(P, mu, label='$\mu_o$ (V&B / B&R)', color='brown', linewidth=2)
    plt.axvline(x=Pb_val, color='red', linestyle='--', label=f'$P_b$ ({Pb_val:.0f} psi)')
    plt.title('Viscosidad del Petróleo ($\mu_o$)', fontsize=12)
    plt.ylabel('$\mu_o$ (cp)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # --- Gráfico 4: Densidad (rho_o) ---
    plt.subplot(3, 2, 4)
    plt.plot(P, rho, label='$\rho_o$ (Standing mod.)', color='purple', linewidth=2)
    plt.axvline(x=Pb_val, color='red', linestyle='--', label=f'$P_b$ ({Pb_val:.0f} psi)')
    plt.title('Densidad del Petróleo ($\rho_o$)', fontsize=12)
    plt.ylabel('Densidad (lb/ft³)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # --- Gráfico 5: Compresibilidad (Co) ---
    plt.subplot(3, 2, 5)
    plt.plot(P, Co * 1e6, label='$C_o$ (Petrosky/Vasquez)', color='orange', linewidth=2)
    plt.axvline(x=Pb_val, color='red', linestyle='--', label=f'$P_b$ ({Pb_val:.0f} psi)')
    plt.axhline(y=0, color='gray', linestyle=':', label='Saturado = 0')
    plt.title('Compresibilidad del Petróleo ($C_o$)', fontsize=12)
    plt.xlabel('Presión (psi)', fontsize=10)
    plt.ylabel('$C_o$ ($\cdot 10^6$) (1/psi)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Perfiles PVT vs. Presión | Yacimiento: {Pr_val:.0f} psi, {Tr_val:.0f} °F, API {API:.1f}', fontsize=16)
    plt.show()

generar_graficos_pvt(P_array, Bo_array, Rs_array, Co_array, rho_array, mu_array, Pb, Pr)

# --- 6. Preparación para Xlwings ---
print("\n--- Datos Generados para Xlwings ---")
print("Columna P: ", P_array)
print("Columna Co: ", Co_array)
print("Columna Rs: ", Rs_array)
print("Columna Bo: ", Bo_array)
print("Columna Rho: ", rho_array)
print("Columna Mu: ", mu_array)