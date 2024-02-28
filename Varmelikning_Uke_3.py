# import numpy as np
# import matplotlib.pyplot as plt 
# import time

# from matplotlib import cm
# from matplotlib.ticker import LinearLocator

# g=0.1
# lengde=20
# tid=20

# X, Y = np.meshgrid(np.arange(0, lengde, 1), np.arange(0, tid, 1))

# u=np.empty((lengde,tid))

# u[5:10,0]=100


# #Euler Eksplisitt
# # for i in range(1, lengde-1):
# #     for j in range(tid-1):
# #         u[i,j+1]=g*u[i+1,j]+g*u[i-1,j]+u[i,j]*(1-2*g)

# #Euler Implisitt
# for i in range(1, lengde-1):
#     for j in range(tid-1):
#         u[i,j+1]=g*u[i+1,j+1]-2*g*u[i,j+1]+g*u[i-1,j+1]+u[i,j]




# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Plot the surface.


# surf = ax.plot_surface(X, Y, u, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
    



# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')


# fig.colorbar(surf, shrink=0.5, aspect=5)



# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

g = 1
lengde = 20
tid = 20

# Create meshgrid for plotting
X, Y = np.meshgrid(np.arange(0, lengde, 1), np.arange(0, tid, 1))

# Initialize arrays for each method
u_explicit = np.empty((lengde, tid))
u_implicit = np.empty((lengde, tid))
u_cn = np.empty((lengde, tid))  # Crank-Nicolson

# Initial condition for all methods
# for u in [u_explicit, u_implicit, u_cn]:
#     u[:, 0] = 0
#     u[5:10, 0] = 100

# Apply sin(x) as initial condition for all methods
x_values = np.arange(0, lengde, 1)
initial_condition = np.sin(x_values) 
for u in [u_explicit, u_implicit, u_cn]:
    u[:, 0] = initial_condition

# Euler Explicit
for j in range(tid - 1):
    for i in range(1, lengde - 1):
        u_explicit[i, j+1] = g*u_explicit[i+1, j] + g*u_explicit[i-1, j] + u_explicit[i, j]*(1 - 2*g)

# Euler Implicit
a = -g
b = 1 + 2*g
c = -g
for j in range(tid - 1):
    A = np.diag([b]*(lengde - 2)) + np.diag([a]*(lengde - 3), -1) + np.diag([c]*(lengde - 3), 1)
    B = u_implicit[1:-1, j]
    u_implicit[1:-1, j+1] = np.linalg.solve(A, B)

# Crank-Nicolson
a_cn = -g / 2
b_cn = 1 + g
c_cn = -g / 2
for j in range(tid - 1):
    A_cn = np.diag([b_cn]*(lengde - 2)) + np.diag([a_cn]*(lengde - 3), -1) + np.diag([c_cn]*(lengde - 3), 1)
    B_cn = np.empty(lengde - 2)
    for i in range(1, lengde - 1):
        B_cn[i - 1] = g/2 * u_cn[i+1, j] + (1 - g) * u_cn[i, j] + g/2 * u_cn[i-1, j]
    u_cn[1:-1, j+1] = np.linalg.solve(A_cn, B_cn)

# Plotting
fig, axs = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 18))

# Plot for Euler Explicit
surf_explicit = axs[0].plot_surface(X, Y, u_explicit, cmap=cm.coolwarm, linewidth=0, antialiased=False)
axs[0].set_title("Euler Explicit")
axs[0].zaxis.set_major_locator(LinearLocator(10))
axs[0].zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf_explicit, ax=axs[0], shrink=0.5, aspect=5)

# Plot for Euler Implicit
surf_implicit = axs[1].plot_surface(X, Y, u_implicit, cmap=cm.coolwarm, linewidth=0, antialiased=False)
axs[1].set_title("Euler Implicit")
axs[1].zaxis.set_major_locator(LinearLocator(10))
axs[1].zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf_implicit, ax=axs[1], shrink=0.5, aspect=5)

# Plot for Crank-Nicolson
surf_cn = axs[2].plot_surface(X, Y, u_cn, cmap=cm.coolwarm, linewidth=0, antialiased=False)
axs[2].set_title("Crank-Nicolson")
axs[2].zaxis.set_major_locator(LinearLocator(10))
axs[2].zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf_cn, ax=axs[2], shrink=0.5, aspect=5)


plt.show()

