import numpy as np
from scipy.optimize import fsolve

def bruggerman_mixing_model(volume_fractions, epsilon_values_real, epsilon_values_imag):
    # Combine real and imaginary parts into complex numbers
    epsilon_values = np.array([complex(real, imag) for real, imag in zip(epsilon_values_real, epsilon_values_imag)])
    
    # Define a function that splits the real and imaginary parts of epsilon_eff
    def equation(x):
        epsilon_eff = complex(x[0], x[1])  # Combine into a complex number
        numerator = 0
        denominator = 0
        for i in range(len(volume_fractions)):
            numerator += (epsilon_values[i] - epsilon_eff) * volume_fractions[i]
            denominator += (epsilon_values[i] + 2 * epsilon_eff) * volume_fractions[i]
        result = numerator / denominator
        return [result.real, result.imag]  # Return real and imaginary parts separately

    # Initial guess for fsolve (real and imaginary parts separately)
    initial_guess = [epsilon_values[0].real, epsilon_values[0].imag]
    
    # Solve for the effective permittivity (real and imaginary parts separately)
    solution = fsolve(equation, initial_guess)
        
    return solution

# Example usage
volume_fractions = [0.3, 0.7]
epsilon_values_real = [2.5, 1.5]
epsilon_values_imag = [0.1, 0.2]

[effective_permittivity, effective_conductivity] = bruggerman_mixing_model(volume_fractions, epsilon_values_real, epsilon_values_imag)

# Extract real and imaginary parts
real_part = effective_permittivity
imaginary_part = effective_conductivity

print("Real Part:", real_part)
print("Imaginary Part:", imaginary_part)
