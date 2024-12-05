# type: ignore
import numpy as np
import matplotlib.pyplot as plt
from util import StateVector

def plot_bloch_sphere(state_vector: StateVector):
    # This code was taken directly from the following link
    # https://stackoverflow.com/a/70445535
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') # type: ignore

    # Make data
    r = 1
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, alpha=0.5, color='linen')

    # plot circular curves over the surface
    arr360 = np.linspace(0, 2 * np.pi, 100)
    z = np.zeros(100)
    x = r * np.sin(arr360)
    y = r * np.cos(arr360)

    ax.plot(x, y, z, color='black', alpha=0.75)
    ax.plot(z, x, y, color='black', alpha=0.75)

    ## add axis lines
    zeros = np.zeros(1000)
    line = np.linspace(-r,r,1000)

    ax.plot(line, zeros, zeros, color='black', alpha=0.75)
    ax.plot(zeros, line, zeros, color='black', alpha=0.75)
    ax.plot(zeros, zeros, line, color='black', alpha=0.75)

    # The Following Code Is My Own
    alpha, beta = state_vector[0], state_vector[1]

    theta = 2 * np.arccos(np.abs(alpha)) # Angle from the z-axis
    phi = np.angle(beta) - np.angle(alpha) # Angle in the xy-plane

    # Convert to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # State vector
    ax.quiver(0, 0, 0, x, y, z, color='blue', linewidth=2, alpha=0.75)

    # Labels for Poles |0> and |1>
    ax.text(0, 0, r + (r/5), r'$|1\rangle$', fontsize=12, ha='center')
    ax.text(0, 0, -r - (r/5), r'$|0\rangle$', fontsize=12, ha='center')

    plt.show()