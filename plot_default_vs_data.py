import numpy as np

# parameters
R1 = 1.25; H1 = 0.8
r_middle_lower = 1.24402  # cap top radius ≈1.16619
r_middle_upper = 2.75  # top of middle frustum / bottom of top
H_middle = 9.89148
r_top = 2.96
H_top = 4.25

# spherical cap volume
V_cap = (np.pi * H1**2 / 3) * (3 * R1 - H1)

# middle frustum full volume
V_middle = (np.pi * H_middle / 3) * (
    r_middle_lower**2 + r_middle_lower * r_middle_upper + r_middle_upper**2
)

# top cylinder (frustum with equal radii) volume
V_top = np.pi * r_top**2 * H_top

print("Spherical cap:", V_cap)
print("Middle frustum:", V_middle)
print("Top section:", V_top)
print("Total:", V_cap + V_middle + V_top)
