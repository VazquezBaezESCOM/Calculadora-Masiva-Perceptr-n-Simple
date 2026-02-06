import numpy as np

class Perceptron:
    def __init__(self, num_entradas, pesos_iniciales, tasa_aprendizaje=1.0):
        self.num_entradas = num_entradas
        self.r = tasa_aprendizaje
        # Usar pesos proporcionados por el usuario (incluyendo bias)
        self.pesos = pesos_iniciales
        print(f"Perceptrón configurado con {num_entradas} entradas + 1 bias")
        print(f"Tasa de aprendizaje: r={self.r}")
        print("-" * 60)
    
    def funcion_activacion(self, x):
        """Función escalón para clasificación binaria"""
        return 1 if x > 0 else 0
    
    def predecir_detallado(self, entrada):
        """Realizar una predicción con cálculo completo detallado"""
        # Añadir bias (siempre 1)
        entrada_con_bias = list(entrada) + [1]
        
        print(f"ENTRADA: {entrada} + bias=1")
        print(f"PESOS: {[f'w{i+1}={w:.4f}' for i, w in enumerate(self.pesos)]}")
        
        # Calcular multiplicaciones paso a paso
        multiplicaciones = []
        suma_ponderada = 0
        
        print("MULTIPLICACIONES:")
        for i, (x, w) in enumerate(zip(entrada_con_bias, self.pesos)):
            multiplicacion = x * w
            multiplicaciones.append(multiplicacion)
            suma_ponderada += multiplicacion
            print(f"  x{i+1} * w{i+1} = {x} * {w:.4f} = {multiplicacion:.4f}")
        
        print(f"SUMA PONDERADA: {' + '.join([f'{m:.4f}' for m in multiplicaciones])} = {suma_ponderada:.4f}")
        
        salida = self.funcion_activacion(suma_ponderada)
        print(f"SALIDA (f({suma_ponderada:.4f})) = {salida}")
        
        return salida, suma_ponderada, entrada_con_bias
    
    def predecir_simple(self, entrada):
        """Predicción simple sin prints para visualización"""
        entrada_con_bias = list(entrada) + [1]
        suma_ponderada = sum(x * w for x, w in zip(entrada_con_bias, self.pesos))
        salida = self.funcion_activacion(suma_ponderada)
        return salida, suma_ponderada
    
    def entrenar_iteracion_detallada(self, X_entrenamiento, y_objetivo):
        """Entrenar por una época completa con cálculo detallado"""
        error_total = 0
        print("=== INICIO DE ÉPOCA ===")
        
        for i, (entrada, objetivo) in enumerate(zip(X_entrenamiento, y_objetivo)):
            print(f"\n--- Patrón {i+1}: {entrada} -> Objetivo: {objetivo} (Clase {objetivo + 1}) ---")
            
            # Realizar predicción con cálculo detallado
            salida, suma_ponderada, entrada_con_bias = self.predecir_detallado(entrada)
            
            # Aplicar criterio específico con explicación detallada
            necesita_actualizacion = False
            print(f"\nAPLICANDO CRITERIO:")
            
            if suma_ponderada >= 0 and objetivo == 0:  # x·w^T ≥ 0 y x ∈ C1 (Clase 1)
                print(f"  CONDICIÓN: x·w^T = {suma_ponderada:.4f} ≥ 0 y x ∈ C1 (Clase 1)")
                print(f"  ACCIÓN: ω - r·x_i (restar)")
                signo = -1
                necesita_actualizacion = True
                
            elif suma_ponderada <= 0 and objetivo == 1:  # x·w^T ≤ 0 y x ∈ C2 (Clase 2)
                print(f"  CONDICIÓN: x·w^T = {suma_ponderada:.4f} ≤ 0 y x ∈ C2 (Clase 2)")
                print(f"  ACCIÓN: ω + r·x_i (sumar)")
                signo = 1
                necesita_actualizacion = True
                
            else:
                print(f"  CONDICIÓN: Predicción correcta - No se requiere actualización")
                signo = 0
            
            # Calcular error para tracking
            error = objetivo - salida
            error_total += abs(error)
            
            if necesita_actualizacion:
                print(f"\nACTUALIZACIÓN DE PESOS (r={self.r}):")
                
                # Actualizar todos los pesos
                pesos_anteriores = self.pesos.copy()
                
                for j in range(len(self.pesos)):
                    delta_w = self.r * signo * entrada_con_bias[j]
                    self.pesos[j] = pesos_anteriores[j] + delta_w
                    
                    if signo > 0:
                        print(f"  w{j+1}: {pesos_anteriores[j]:.4f} + {self.r:.4f} * {entrada_con_bias[j]} = {self.pesos[j]:.4f}")
                    else:
                        print(f"  w{j+1}: {pesos_anteriores[j]:.4f} - {self.r:.4f} * {entrada_con_bias[j]} = {self.pesos[j]:.4f}")
            else:
                print("✓ No se actualizan pesos - Clasificación correcta")
            
            print(f"Pesos actualizados: {[f'w{i+1}={w:.4f}' for i, w in enumerate(self.pesos)]}")
        
        print(f"\n=== FIN DE ÉPOCA - Error total: {error_total} ===")
        return error_total
    
    def entrenar_iteracion_eficiente(self, X_entrenamiento, y_objetivo):
        """Entrenar por una época completa de manera eficiente (sin prints detallados)"""
        error_total = 0
        
        for entrada, objetivo in zip(X_entrenamiento, y_objetivo):
            # Realizar predicción simple
            entrada_con_bias = list(entrada) + [1]
            suma_ponderada = sum(x * w for x, w in zip(entrada_con_bias, self.pesos))
            salida = self.funcion_activacion(suma_ponderada)
            
            # Aplicar criterio específico
            necesita_actualizacion = False
            
            if suma_ponderada >= 0 and objetivo == 0:  # x·w^T ≥ 0 y x ∈ C1 (Clase 1)
                signo = -1
                necesita_actualizacion = True
                
            elif suma_ponderada <= 0 and objetivo == 1:  # x·w^T ≤ 0 y x ∈ C2 (Clase 2)
                signo = 1
                necesita_actualizacion = True
            
            # Calcular error para tracking
            error = objetivo - salida
            error_total += abs(error)
            
            if necesita_actualizacion:
                # Actualizar pesos
                for j in range(len(self.pesos)):
                    self.pesos[j] += self.r * signo * entrada_con_bias[j]
        
        return error_total
    
    def entrenar(self, X_entrenamiento, y_objetivo, iterations_max=3000, detallado=True):
        """Entrenar el perceptrón hasta convergencia"""
        print("🚀 ENTRENANDO PERCEPTRÓN")
        print("CRITERIO:")
        print("  Si x·w^T ≥ 0 y x ∈ C1 (0) → ω - r·x_i")
        print("  Si x·w^T ≤ 0 y x ∈ C2 (1) → ω + r·x_i")
        print("Mapeo: 0 = Clase 1, 1 = Clase 2")
        print("=" * 60)
        
        for iteration in range(iterations_max):
            print(f"\n🔄 Iteración {iteration + 1}:")
            
            if detallado:
                error_total = self.entrenar_iteracion_detallada(X_entrenamiento, y_objetivo)
            else:
                error_total = self.entrenar_iteracion_eficiente(X_entrenamiento, y_objetivo)
                print(f"Error total: {error_total}")
            
            if error_total == 0:
                print(f"\n🎯 ENTRENAMIENTO COMPLETADO en iteración {iteration + 1}")
                return iteration + 1
            else:
                print(f"\n↻ Continuando entrenamiento...")
                print("=" * 60)
        
        print(f"\n⚠️  Entrenamiento alcanzó el máximo de {iterations_max} iteraciones")
        return iterations_max

    def mostrar_ecuacion(self):
        """Mostrar la ecuación del perceptrón"""
        ecuacion = " + ".join([f"{w:.4f}*x{i+1}" for i, w in enumerate(self.pesos[:-1])])
        ecuacion += f" + {self.pesos[-1]:.4f} = 0"
        return ecuacion

def generar_patrones_binarios(n):
    """Generar todos los patrones binarios de n bits"""
    if n <= 0:
        return []
    
    # Generar todas las combinaciones binarias
    patrones = []
    for i in range(2**n):
        # Convertir a binario y rellenar con ceros a la izquierda
        binario = bin(i)[2:].zfill(n)
        patron = [int(bit) for bit in binario]
        patrones.append(patron)
    
    return patrones

def mostrar_patrones(patrones):
    """Mostrar los patrones generados"""
    print(f"\nSe generaron {len(patrones)} patrones binarios:")
    for i, patron in enumerate(patrones):
        binario = ''.join(str(bit) for bit in patron)
        print(f"  Patrón {i+1}: {patron} (binario: {binario})")

def solicitar_pesos(num_entradas):
    """Solicitar al usuario los pesos iniciales"""
    print(f"\nINGRESE LOS PESOS INICIALES PARA LAS {num_entradas} ENTRADAS + BIAS")
    pesos = []
    
    # Solicitar pesos para las entradas
    for i in range(num_entradas):
        while True:
            try:
                peso = float(input(f"Peso w{i+1} para entrada x{i+1}: "))
                pesos.append(peso)
                break
            except ValueError:
                print("Por favor ingrese un número válido.")
    
    # Solicitar peso para el bias
    while True:
        try:
            peso_bias = float(input(f"Peso w{num_entradas+1} para bias: "))
            pesos.append(peso_bias)
            break
        except ValueError:
            print("Por favor ingrese un número válido.")
    
    return pesos

def main():
    """Función principal del programa"""
    print("🧠 PERCEPTRÓN - CALCULADORA INTERACTIVA")
    print("=" * 60)
    
    # Solicitar número de entradas al usuario
    while True:
        try:
            num_entradas = int(input("\nIngrese el número de entradas (1-10): "))
            if 1 <= num_entradas <= 10:
                break
            else:
                print("Por favor ingrese un número entre 1 y 10.")
        except ValueError:
            print("Por favor ingrese un número válido.")
    
    # Solicitar pesos iniciales al usuario
    pesos_iniciales = solicitar_pesos(num_entradas)
    
    # Solicitar tasa de aprendizaje
    while True:
        try:
            tasa_aprendizaje = float(input("\nTasa de aprendizaje (r): "))
            if tasa_aprendizaje > 0:
                break
            else:
                print("La tasa de aprendizaje debe ser mayor que 0.")
        except ValueError:
            print("Por favor ingrese un número válido.")
    
    print("\n" + "=" * 60)
    
    # Generar patrones binarios automáticamente
    patrones = generar_patrones_binarios(num_entradas)
    mostrar_patrones(patrones)
    
    # Solicitar valores objetivo para cada patrón
    print(f"\nINGRESE LOS VALORES DE y_objetivo PARA LOS {len(patrones)} PATRONES")
    print("(0 para Clase 1, 1 para Clase 2)")
    
    y_objetivo = []
    for i, patron in enumerate(patrones):
        binario = ''.join(str(bit) for bit in patron)
        while True:
            try:
                valor = int(input(f"Patrón {i+1} ({binario}): "))
                if valor in [0, 1]:
                    y_objetivo.append(valor)
                    break
                else:
                    print("Por favor ingrese 0 o 1.")
            except ValueError:
                print("Por favor ingrese 0 o 1.")
    
    print(f"\nValores objetivo ingresados: {y_objetivo}")
    
    # Crear perceptrón con los pesos proporcionados por el usuario
    perceptron = Perceptron(num_entradas=num_entradas, 
                           pesos_iniciales=pesos_iniciales, 
                           tasa_aprendizaje=tasa_aprendizaje)
    
    # Mostrar pesos iniciales
    print(f"\nPesos iniciales asignados: {[f'w{i+1}={w:.4f}' for i, w in enumerate(perceptron.pesos)]}")
    
    # Entrenar con cálculo detallado
    iteraciones_necesarias = perceptron.entrenar(patrones, y_objetivo, detallado=True)
    
    # Probar el perceptrón entrenado
    print("\n" + "=" * 60)
    print("🧪 PRUEBA FINAL DEL PERCEPTRÓN ENTRENADO")
    print("=" * 60)
    
    print("RESULTADOS FINALES:")
    print("-" * 50)
    correctos = 0
    total = len(patrones)
    
    for i, (patron, objetivo) in enumerate(zip(patrones, y_objetivo)):
        # Usar predicción detallada para mostrar cálculos completos
        salida, suma, _ = perceptron.predecir_detallado(patron)
        binario = ''.join(str(bit) for bit in patron)
        clase_predicha = f"Clase {salida + 1}"
        clase_esperada = f"Clase {objetivo + 1}"
        resultado = "✓ CORRECTO" if salida == objetivo else "✗ INCORRECTO"
        
        if salida == objetivo:
            correctos += 1
        
        print(f"Entrada {binario} -> Pred: {salida} ({clase_predicha}) | Esperado: {objetivo} ({clase_esperada}) {resultado}")
        print("-" * 40)
    
    precision = (correctos / total) * 100
    print(f"\n📈 RESUMEN FINAL:")
    print(f"Iteraciones necesarias: {iteraciones_necesarias}")
    print(f"Precisión: {correctos}/{total} = {precision:.1f}%")
    print(f"Pesos finales: {[f'w{i+1}={w:.4f}' for i, w in enumerate(perceptron.pesos)]}")
    
    # Mostrar la ecuación final
    print(f"\n📐 ECUACIÓN FINAL DEL PERCEPTRÓN:")
    ecuacion_final = perceptron.mostrar_ecuacion()
    print(ecuacion_final)
    
    # Información adicional
    print(f"\n🔧 INFORMACIÓN ADICIONAL:")
    print(f"Número de entradas: {num_entradas}")
    print(f"Número de patrones: {len(patrones)}")
    print(f"Tasa de aprendizaje: {tasa_aprendizaje}")
    
    # Mostrar análisis de clases
    clase_1 = y_objetivo.count(0)
    clase_2 = y_objetivo.count(1)
    print(f"Clase 1: {clase_1} patrones")
    print(f"Clase 2: {clase_2} patrones")
    
    # Mostrar pesos iniciales vs finales
    print(f"\n🔁 COMPARACIÓN PESOS INICIALES VS FINALES:")
    for i, (inicial, final) in enumerate(zip(pesos_iniciales, perceptron.pesos)):
        cambio = final - inicial
        print(f"  w{i+1}: {inicial:.4f} → {final:.4f} (Δ = {cambio:+.4f})")

# Ejecutar el programa
if __name__ == "__main__":
    main()